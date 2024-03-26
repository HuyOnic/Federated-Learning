import matplotlib.pyplot as plt


def visualize(data_loaders, num_clients:int):
    fig,ax = plt.subplots(nrows=5,ncols=10, figsize=(40,30))
    plt.subplots_adjust(wspace=.4,hspace=.4)
    indx = 0
    indy = 0
    for client in range(num_clients):
        labels =[]
        for i, batch in enumerate(data_loaders[client]):
            labels.append(batch[1])
        flatten_labels = [item for sub_list in labels for item in sub_list]

        ax[indx,indy].hist(flatten_labels)
        n = client+1
        ax[indx,indy].set_title(f'Client {n}')
        indy+=1
        if indy == 10:
            indx+=1
            indy=0
    plt.show()






