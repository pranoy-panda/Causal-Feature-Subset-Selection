# importing local libraries
from config import *
from dataset_file import *
from utils import *
from models_file import BaseModel, gumbel_selector

print('1. Training the Basemodel...... \n') 
bb_model = BaseModel()
LossFunc_basemodel = torch.nn.CrossEntropyLoss(size_average = True)
optimizer_basemodel = torch.optim.Adam(bb_model.parameters(),lr = lr_basemodel) 

# train the basemodel 
train_basemodel(trainloader,
               bb_model,
               LossFunc_basemodel,
               optimizer_basemodel,
               num_epochs_basemodel,
               kwargs['batch_size'])

# testing the model on held-out validation dataset
test_basemodel(valloader,bb_model)
print('Basemodel trained! \n')

##############################################################

print('2. Starting main feature selection algorithm......... \n')

'''
generate images with random patches selected for calculating the ACE metric
'''
imgs_with_random_patch = generate_imgs_with_random_patch(valloader,num_init)     

# training loop where we run the experiments for multiple times and report the 
# mean and standard deviation of the metrics ph_acc and ICE.
for iter_num in range(num_init):
  # intantiating the gumbel_selector or in other words initializing the explainer's weights
  selector = gumbel_selector()
  #optimizer
  optimizer = torch.optim.Adam(selector.parameters(),lr = lr)
  # variable for keeping track of best ph_acc across different iterations 
  best_iter_val_acc = 0

  # training loop
  for epoch in range(num_epochs):  
      running_loss = 0
      for i, data in enumerate(trainloader, 0):
          # get the inputs
          X, Y = data
          batch_size = X.size(0)
          # zero the parameter gradients
          optimizer.zero_grad()
          # 1: get the logits from construct_gumbel_selector()
          logits = selector.forward(X)
          # 2: get a subset of the features(encoded in a binary vector) by using the gumbel-softmax trick      
          selected_subset = sample_concrete(tau,k,logits,train=True) # get S_bar from explainer
          # 3: reshape selected_subset to the size M x M i.e. the size of the patch or superpixel
          selected_subset = torch.reshape(selected_subset,(batch_size,M,M))
          selected_subset = torch.unsqueeze(selected_subset,dim=1)
          # 4: upsampling the selection map
          upsample_op = nn.Upsample(scale_factor=N, mode='nearest')
          v = upsample_op(selected_subset)
          # 5: X_Sbar = elementwise_multiply(X,v); compute f_{bb}(X_Sbar)
          X_Sbar = torch.mul(X,v) # output shape will be [batch_size,1,M*N,M*N]
          
          f_xsbar = F.softmax(bb_model(X_Sbar)) # f_xs stores p(y|xs)
          with torch.no_grad():
            f_x =  F.softmax(bb_model(X)) # f_x stores p(y|x)          

          # optimization function
          loss = custom_loss(f_xsbar,f_x,batch_size)

          loss.backward()
          optimizer.step()
        
          running_loss+=loss.item() # average loss per sample

      val_acc,ICE = metrics(selector,M,N,iter_num,valloader,imgs_with_random_patch,bb_model)
      if best_iter_val_acc<val_acc:
        best_iter_val_acc = val_acc
        corresponding_ice = ICE
      if best_val_acc<val_acc:
        best_val_acc = val_acc
        best_selector = selector  
        torch.save(selector, 'selector.pt')              
      # print loss and validation accuracy at the end of each epoch 
      print('\nInitialization Number %d-> epoch: %d, average loss: %.3f, val_acc: %.3f, ICE: %.3f \n' %(iter_num+1, epoch + 1,running_loss/i, val_acc, ICE))

  val_acc_list.append(best_iter_val_acc)
  ice_list.append(corresponding_ice)
  print('Best val acc for iteration %d : %.3f'%(iter_num+1,best_iter_val_acc))   
      
print('best validation accuracy: %.2f'%(best_val_acc)) 
print('mean ph acc: %.3f'%(np.mean(val_acc_list)),', std dev: %.3f '%(np.std(val_acc_list))) 
print('mean ICE: %.3f'%(np.mean(ice_list)),', std dev: %.3f '%(np.std(ice_list))) 

print('\nCausal Feature Selector trained! \n')
######################################################

print('3. Visualize the results of the trained selector model .....\n')
visualize_results(gumbel_selector,valloader,bb_model)
