import numpy as np

def get_pool_predictions(trained_net, pool_loader,device, return_labels = False):
  trained_net.eval()

  yhat = []
  labels_list = []
  for (data,labels) in pool_loader:
    pred = trained_net(data.to(device).float())
    yhat.append( pred.to("cpu").detach().numpy())
    labels_list.append(labels)
  
  predictions = np.concatenate(yhat)
  labels_list = np.concatenate(labels_list)
  
  if return_labels:
    return predictions,labels_list
  else:
    return predictions
