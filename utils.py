# importing local libraries
from config import *

def break_point():
  sys.exit()

def visualize_results(gumbel_selector,valloader,bb_model):
  selector = gumbel_selector()
  selector = torch.load('selector.pt')

  selector.eval()

  correct_count, all_count = 0, 0
  ICE = 0
  for images,labels in valloader:
    for i in range(len(images)):
      img = images[i].unsqueeze(0)
      xs,v = generate_xs(img,selector,M,N)
      with torch.no_grad():
          # get the augmented image(from val. dataset)
          out_xs = F.softmax(bb_model(xs),dim=1)
          out_x = F.softmax(bb_model(img),dim=1)
      pred_label = torch.argmax(out_xs)
      true_label = torch.argmax(out_x)
      # post hoc accuracy calc.
      if(true_label == pred_label):
        correct_count += 1
      all_count += 1     

      # some visualization
      if all_count%50==0: #%50
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 10))
  
        ax1.imshow(img.numpy().squeeze(), cmap='gray');
        # using yticks and xticks to draw out the regions depicting the patches
        ax1.set_yticks(np.arange(0,28,4), minor=False)
        ax1.yaxis.grid(True, which='major')
        ax1.set_xticks(np.arange(0,28,4), minor=False)
        ax1.xaxis.grid(True, which='major')
        ax1.set_title('class predicted by image: '+str(true_label))
  
        #ax2.imshow(xs.numpy().squeeze(), cmap='gray');
        ax2.imshow(img.numpy().squeeze(),cmap='gray')
        ax2.imshow(v[0][0].cpu(),'copper', alpha=0.5)
        ax2.set_title('X_S (class predicted by subset: '+str(pred_label)) 

      plt.show()  
      
  print("Number Of Images Tested =", all_count)
  print("\nModel Accuracy(with X_S as input) = %.3f" %(correct_count/all_count))

'''
Given an instance X and the selector network, this function returns X_S,X_Sbar and S_bar
X_S: augmented X, where the un-selected patches are masked out(here, replaced by zero) 
X_Sbar: augmented X, where the selected patches are masked out(here, replaced by zero)
S_bar: a map/2D matrix where the un-selected patches are set to 1 and selected patches are set to 0
'''
def generate_xs(X,selector,M,N): # M x M is the size of the patch, and M*N x M*N is the size of the instance X
  batch_size = X.shape[0]
  # 1: get the logits from the selector for instance X
  with torch.no_grad():
    logits = selector.forward(X) # shape is (bs,M*M), where M is the patch size
  # 2: get a subset of the features(encoded in a binary vector) by using the gumbel-softmax trick
  selected_subset = sample_concrete(tau,k,logits,train=False)# get S_bar from explainer
  # 3: reshape selected_subset to the size M x M i.e. the size selection map
  selected_subset = torch.reshape(selected_subset,(batch_size,M,M))
  selected_subset = torch.unsqueeze(selected_subset,dim=1)# S_bar
  selected_subset_inverted = torch.abs(selected_subset-1)# getting S from S_bar
  # 4: upsample the selection map
  upsample_op = nn.Upsample(scale_factor=N, mode='nearest')
  v = upsample_op(selected_subset_inverted)
  # 5: X_S = elementwise_multiply(X,v); compute f_{bb}(X_S)
  X_S = torch.mul(X,v) # output shape will be [bs,1,M*N,M*N] 
  #X_Sbar = torch.mul(X,v_bar)
  return X_S,v#,X_Sbar

'''
This function calculates thes two metrics for evaluating the explanations
1. post hoc accuracy
2. average ICE: (1/batch_size)*( p(y=c/xs) - p(y=c/x') ), here c is class 
   predicted by basemodel and x' is the image where k patches are randomly selected from x are present, rest all patches are null
'''
def metrics(selector,M,N,init_num,valloader,imgs_with_random_patch,bb_model):
  correct_count, all_count = 0, 0
  ICE = 0
  for images,labels in valloader:
    for i in range(len(images)):
      img = images[i].unsqueeze(0)
      xs,v = generate_xs(img,selector,M,N)
      xprime = torch.Tensor(imgs_with_random_patch[init_num][all_count]).unsqueeze(0).unsqueeze(0)
      with torch.no_grad():
          # get the augmented image(from val. dataset)
          out_xs = F.softmax(bb_model(xs),dim=1)
          out_xprime = F.softmax(bb_model(xprime),dim=1)
          out_x = F.softmax(bb_model(img),dim=1)
      pred_label = torch.argmax(out_xs)
      true_label = torch.argmax(out_x)
      # post hoc accuracy calc.
      if(true_label == pred_label):
        correct_count += 1
      #ICE calc.    
      ICE+=out_xs[0][true_label]-out_xprime[0][true_label]
      all_count += 1

  ph_acc = (correct_count/all_count)
  ACE=ICE/all_count      

  return ph_acc,ACE 

'''
This function samples from a concrete distribution during training and while inference, it gives the indices of the top k logits
'''
def sample_concrete(tau,k,logits,train=True):
  # input logits dimension: [batch_size,1,d]
  logits = logits.unsqueeze(1)
  d = logits.shape[2]
  batch_size = logits.shape[0]  
  if train == True:
    softmax = nn.Softmax() # defining the softmax operator
    unif_shape = [batch_size,k,d] # shape for uniform distribution, notice there is k. Reason: we have to sample k times for k features
    uniform = (1 - 0) * torch.rand(unif_shape) # generating vector of shape "unif_shape", uniformly random numbers in the interval [0,1)
    gumbel = - torch.log(-torch.log(uniform)) # generating gumbel noise/variables
    noisy_logits = (gumbel + logits)/tau # perturbed logits(perturbed by gumbel noise and temperature coeff. tau)
    samples = softmax(noisy_logits) # sampling from softmax
    samples,_ = torch.max(samples, axis = 1) 
    return samples
  else:  
    logits = torch.reshape(logits,[-1, d]) 
    discrete_logits = torch.zeros(logits.shape[1])
    vals,ind = torch.topk(logits,k)
    discrete_logits[ind[0]]=1    
    discrete_logits = discrete_logits.type(torch.float32) # change type to float32
    discrete_logits = torch.unsqueeze(discrete_logits,dim=0)
    return discrete_logits   

'''
custom loss function that is for our objective function(similar to categorical cross entropy function)
p_y_xs is the p(y|xs) or f_{bb}(xs)
p_y_xs is the p(y|x) or f_{bb}(x)
'''
def custom_loss(p_y_xs,p_y_x,batch_size):
    loss= torch.mean(torch.sum(p_y_x.view(batch_size, -1) * torch.log(p_y_xs.view(batch_size, -1)), dim=1))
    return loss

def generate_imgs_with_random_patch(valloader,num_init):
  num_validation_images = len(valloader.dataset)
  img_size = valloader.dataset.data.shape[1]
  imgs_with_random_patch = np.zeros((num_init,num_validation_images,img_size,img_size))
  img_count = 0 
  for i in range(num_init):
    for images,_ in valloader:
      for j in range(len(images)):
        patch_selection_map = np.zeros((M*M))
        patch_selection_map[:num_patches] = 1
        np.random.shuffle(patch_selection_map) # random permutation of the above created array with 'num_patches' ones
        patch_selection_map = patch_selection_map.reshape(M,M)
        patch_selection_map = np.kron(patch_selection_map, np.ones((N,N))) # upsampled to size (M*N,M*N)   
        imgs_with_random_patch[i][img_count] = np.multiply(patch_selection_map,images[j])
        img_count+=1 # updating image count
    img_count = 0  

    return imgs_with_random_patch

def train_basemodel(trainloader,bb_model,LossFunc,optimizer,num_epochs,batch_size):
  #training loop
  for epoch in range(num_epochs):
    with tqdm(trainloader, unit="batch") as tepoch:
      for data, target in tepoch:
        tepoch.set_description("Epoch "+str(epoch))
        
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = bb_model(data)
        loss = LossFunc(outputs, target)

        predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == target).sum().item()
        accuracy = correct / batch_size
        
        loss.backward()
        optimizer.step()

        tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

  # uncomment to save the model
  torch.save(bb_model, 'mnist_model.pt') 

def test_basemodel(valloader,bb_model):
  # testing the black box model performance on the entire validation dataset
  correct_count, all_count = 0, 0
  for images,labels in valloader:
    for i in range(len(labels)):
      img = images[i]
      img = img.unsqueeze(0)
      with torch.no_grad():
          out = bb_model(img)

      pred_label = torch.argmax(out)
      true_label = labels.numpy()[i]
      if(true_label == pred_label):
        correct_count += 1
      all_count += 1

  print("Number Of Images Tested =", all_count)
  print("Model Accuracy =", (correct_count/all_count)) 