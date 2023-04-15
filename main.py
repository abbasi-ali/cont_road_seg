import torch
import numpy as np 
from torch.utils.data import DataLoader
from models import * 
import os 
from torchvision import transforms
from data_utils import *
import matplotlib.pyplot as plt
from vgg import VGGNet
from utils import *
from ewc_utils import * 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

@torch.no_grad()
def eval(model, ds_tst, verbose=True, end_eval=None, multihead=False, t_cnt=None):
    bs = 128
    dl_tst = DataLoader(ds_tst, batch_size=bs, shuffle=True)

    model.eval()
    with torch.no_grad():
        iou = []
        for i, data in enumerate(dl_tst):
            image, mask = data
            image, mask = image.to(device), mask.squeeze().to(device)
            output = model(image)
            if multihead:
                output = output[t_cnt]

            pred = output.argmax(dim=1)

            
            # if i == 0 and verbose:
            #     f, axarr = plt.subplots(3, 4, figsize=(30, 15))
            #     mask = mask.unsqueeze(1)
            #     pred = pred.unsqueeze(1)
                
            #     for j in range(4):
            #         axarr[0, j].imshow(image[j].permute(1, 2, 0).cpu().numpy())
            #         axarr[1, j].imshow(mask[j].permute(1, 2, 0)[:, :, 0].cpu().numpy(), vmin=0, vmax=33, cmap='jet')
            #         axarr[2, j].imshow(pred[j].permute(1, 2, 0)[:, :, 0].cpu().numpy(), vmin=0, vmax=33, cmap='jet')
                
            #     plt.savefig('pred_mask_weathers.png')

            
            iou.append(calculate_iou(pred, mask, 34))
            
            if i == end_eval:
                break
        
        if verbose:
            print('Mean IoU: {}'.format(np.mean(iou)))


    model.train()
    return np.mean(iou)


def train_one_task(data_root, save_name, loss='CE'):
    # model = Res18BB(num_classes=34).to(device)
    # model = FSUNet(in_c=3, out_c=64, num_classes=34).to(device)

    vgg_model = VGGNet(model='vgg19', requires_grad=True)
    model = FCN8s(pretrained_net=vgg_model, n_class=34).to(device)

    nepoch = 10
    bs = 16
    lr=1e-3
    save_every = 1
    eval_every = 20

    transform = transforms.Compose([
        transforms.Resize((256, 512))
    ])
    
    home_path = os.path.expanduser('~')
    data_root = os.path.join(home_path, 'data', data_root)
    label_root = os.path.join(home_path, 'data', 'gtFine')

    ds_train = CityScapeDataset(data_root=data_root, label_root=label_root, transform=transform, split='train')
    ds_test = CityScapeDataset(data_root=data_root, label_root=label_root, transform=transform, split='test')

    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)

    if loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif loss == 'focal':
        criterion = FocalLoss(num_classes=34, gamma=4).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_ = []

    for epoch in range(nepoch):
        for i, data in enumerate(dl_train):
            image, mask = data
            image, mask = image.to(device), mask.squeeze().to(device)           

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            loss_.append(loss.item())

            if (i+1) % eval_every == 0 or (i+1) == len(dl_train):
                __ = eval(model, ds_test, verbose=True)

            if (i+1) % save_every == 0:
                print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
                
                plt.plot(loss_)
                plt.savefig(f'{save_name}_training_loss.png')
                plt.close()

                torch.save(model.state_dict(), f'{save_name}_model.pth')

                fig, ax = plt.subplots(3, 5, figsize=(20, 10))
                #first row: input images second row: ground truth third row: predicted
                #title accordingly
                for i in range(5):
                    ax[0, i].imshow(image[i].cpu().detach().permute(1, 2, 0))
                    ax[0, i].set_title('input')

                    ax[1, i].imshow(mask[i].cpu().detach(), cmap='viridis', vmin=0, vmax=33)
                    ax[1, i].set_title('ground truth')

                    ax[2, i].imshow(output[i].cpu().detach().argmax(dim=0), cmap='viridis', vmin=0, vmax=33)
                    ax[2, i].set_title('predicted')

                plt.savefig(f'{save_name}_training_results.png')
                plt.close()

def train_cont_cities_vanilla(data_root, save_name, loss='CE', seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(f'./{save_name}'):
        os.makedirs(save_name)

    save_path = os.path.join(os.getcwd(), save_name)

    vgg_model = VGGNet(model='vgg19', requires_grad=True)
    model = FCN8s(pretrained_net=vgg_model, n_class=34).to(device)

    nepoch = 20
    bs = 16
    lr=1e-3
    save_every = 5
    eval_every = 10
    task_num = 6

    transform = transforms.Compose([
        transforms.Resize((256, 512))
    ])
    
    home_path = os.path.expanduser('~')
    data_root = os.path.join(home_path, 'data', data_root)
    label_root = os.path.join(home_path, 'data', 'gtFine')

    city_splits = get_city_task_splits(data_root, task_num)
    # city_splits = get_city_task_splits(data_root, 18) #one city per task

    print(f'Task splits are {city_splits}')

    if loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif loss == 'focal':
        criterion = FocalLoss(num_classes=34, gamma=3, alpha=1).to(device)

    acc_mat = np.zeros((task_num, task_num))

    train_datasets, test_datasets = [], []

    for t_cnt in range(task_num):
        ds_train = CityScapeDatasetWithinCitySplit(data_root=data_root, label_root=label_root, 
                                    transform=transform, split='train', city_names=city_splits[t_cnt], seed=seed)
        
        ds_test = CityScapeDatasetWithinCitySplit(data_root=data_root, label_root=label_root, transform=transform, 
                                   split='test', city_names=city_splits[t_cnt], seed=seed)
        
        train_datasets.append(ds_train)
        test_datasets.append(ds_test)

    
    for t_cnt in range(task_num):
        dl_train = DataLoader(train_datasets[t_cnt], batch_size=bs, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_ = []

        for epoch in range(nepoch):
            for i, data in enumerate(dl_train):
                image, mask = data
                image, mask = image.to(device), mask.squeeze().to(device)           

                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, mask)
                loss.backward()
                optimizer.step()

                loss_.append(loss.item())

                if (i+1) % eval_every == 0 or (i+1) == len(dl_train):
                    __ = eval(model, test_datasets[t_cnt], verbose=True)
                    torch.save(model.state_dict(), f'{save_path}/{save_name}_model_t{t_cnt}.pth')

                if (i+1) % save_every == 0 or (i+1) == len(dl_train):
                    print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
                    
                    plt.plot(loss_)
                    plt.savefig(f'{save_path}/{save_name}_training_loss.png')
                    plt.close()

                    fig, ax = plt.subplots(3, 5, figsize=(20, 10))
                    #first row: input images second row: ground truth third row: predicted
                    #title accordingly
                    for i in range(5):
                        ax[0, i].imshow(image[i].cpu().detach().permute(1, 2, 0))
                        ax[0, i].set_title('input')

                        ax[1, i].imshow(mask[i].cpu().detach(), cmap='viridis', vmin=0, vmax=33)
                        ax[1, i].set_title('ground truth')

                        ax[2, i].imshow(output[i].cpu().detach().argmax(dim=0), cmap='viridis', vmin=0, vmax=33)
                        ax[2, i].set_title('predicted')

                    plt.savefig(f'{save_path}/{save_name}_training_results.png')
                    plt.close()

        # if t_cnt == 0:
        #     for name, param in model.pretrained_net.named_parameters():
        #         param.requires_grad = False
        
        for eval_cnt in range(task_num):            
            acc_mat[t_cnt, eval_cnt]  = eval(model, test_datasets[eval_cnt], verbose=False)

        np.save(f'{save_path}/{save_name}_acc_mat.npy', acc_mat)
        
        with np.printoptions(precision=2):
            print(acc_mat)

    print(f'Task splits are {city_splits}')

def train_cont_weathers_vanilla(save_name, loss='CE', seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(f'./{save_name}'):
        os.makedirs(save_name)

    save_path = os.path.join(os.getcwd(), save_name)

    vgg_model = VGGNet(model='vgg19', requires_grad=True)
    model = FCN8s(pretrained_net=vgg_model, n_class=34).to(device)

    nepoch = 5
    bs = 32
    lr=1e-3
    save_every = 1
    eval_every = 10
    task_num = 3

    transform = transforms.Compose([
        transforms.Resize((256, 512))
    ])
    
    home_path = os.path.expanduser('~')
    data_root = os.path.join(home_path, 'data')
    label_root = os.path.join(home_path, 'data', 'gtFine')

    if loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif loss == 'focal':
        criterion = FocalLoss(num_classes=34, gamma=3, alpha=1).to(device)

    acc_mat = np.zeros((task_num, task_num))

    train_city_names = ['aachen', 'bochum', 'bremen', 'cologne', 
                  'dusseldorf', 'erfurt', 'hanover', 'jena', 'weimar']
    
    test_city_names = ['lindau', 'munster']
    
    task_data_roots = ['leftImg8bit', 'leftImg8bit_foggy', 'leftImg8bit_rain']
    # task_data_roots = ['leftImg8bit_foggy', 'leftImg8bit_rain', 'leftImg8bit']
    # task_data_roots = ['leftImg8bit_rain', 'leftImg8bit_foggy', 'leftImg8bit']

    train_datasets, test_datasets = [], []

    for t_cnt in range(task_num):
        task_path = os.path.join(data_root, task_data_roots[t_cnt])

        if 'rain' in task_path:
            ds_train = CityScapeDatasetRain(data_root=task_path, label_root=label_root, 
                                        transform=transform, split='train', seed=seed)
            ds_test = CityScapeDatasetRain(data_root=task_path, label_root=label_root, transform=transform, 
                                    split='test', seed=seed)
        else:
            ds_train = CityScapeDataset(data_root=task_path, label_root=label_root, 
                                        transform=transform, split='train', seed=seed)
            ds_test = CityScapeDataset(data_root=task_path, label_root=label_root, transform=transform, 
                                    split='test', seed=seed)
            
        train_datasets.append(ds_train)
        test_datasets.append(ds_test)


    for t_cnt in range(task_num):
        dl_train = DataLoader(train_datasets[t_cnt], batch_size=bs, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_ = []

        print(f'Current task is {task_data_roots[t_cnt]}, Task number is {t_cnt}')

        for epoch in range(nepoch):
            for i, data in enumerate(dl_train):
                image, mask = data
                image, mask = image.to(device), mask.squeeze().to(device)           

                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, mask)
                loss.backward()
                optimizer.step()

                loss_.append(loss.item())
                
                if (i+1) % eval_every == 0 or (i+1) == len(dl_train):
                    __ = eval(model, test_datasets[t_cnt], verbose=True, end_eval=5)
                    torch.save(model.state_dict(), f'{save_path}/{save_name}_model_t{t_cnt}.pth')

                if (i+1) % save_every == 0 or (i+1) == len(dl_train):
                    print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
                    
                    plt.plot(loss_)
                    plt.savefig(f'{save_path}/{save_name}_training_loss.png')
                    plt.close()

                    fig, ax = plt.subplots(3, 5, figsize=(20, 10))
                    #first row: input images second row: ground truth third row: predicted
                    #title accordingly
                    for i in range(5):
                        ax[0, i].imshow(image[i].cpu().detach().permute(1, 2, 0))
                        ax[0, i].set_title('input')

                        ax[1, i].imshow(mask[i].cpu().detach(), cmap='viridis', vmin=0, vmax=33)
                        ax[1, i].set_title('ground truth')

                        ax[2, i].imshow(output[i].cpu().detach().argmax(dim=0), cmap='viridis', vmin=0, vmax=33)
                        ax[2, i].set_title('predicted')

                    plt.savefig(f'{save_path}/{save_name}_training_results.png')
                    plt.close()


        if t_cnt == 0:
            for name, param in model.pretrained_net.named_parameters():
                param.requires_grad = False
        
        for eval_cnt in range(task_num):            
            acc_mat[t_cnt, eval_cnt]  = eval(model, test_datasets[eval_cnt], verbose=False)

        np.save(f'{save_path}/{save_name}_acc_mat.npy', acc_mat)
        
        with np.printoptions(precision=2):
            print(acc_mat)

    # print(f'Task splits are {city_splits}')



def train_cont_cities_ewc(data_root, save_name, loss_type='CE', seed=0, multihead=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    weight_ewc = 5e3
    task_num = 6

    if not os.path.exists(f'./{save_name}'):
        os.makedirs(save_name)

    save_path = os.path.join(os.getcwd(), save_name)

    vgg_model = VGGNet(model='vgg19', requires_grad=True)

    if multihead:
        model = FCN8sMH(pretrained_net=vgg_model, n_class=34, n_tasks=task_num).to(device)
    else:
        model = FCN8s(pretrained_net=vgg_model, n_class=34).to(device)

    nepoch = 20
    bs = 16
    lr=1e-3
    save_every = 5
    eval_every = 10
    

    transform = transforms.Compose([
        transforms.Resize((256, 512))
    ])
    
    home_path = os.path.expanduser('~')
    data_root = os.path.join(home_path, 'data', data_root)
    label_root = os.path.join(home_path, 'data', 'gtFine')

    # city_splits = get_city_task_splits(data_root, task_num)
    city_splits = get_city_task_splits(data_root, 18) #one city per task

    print(f'Task splits are {city_splits}')

    if loss_type == 'CE':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif loss_type == 'focal':
        criterion = FocalLoss(num_classes=34, gamma=3, alpha=1).to(device)

    acc_mat = np.zeros((task_num, task_num))

    train_datasets, test_datasets = [], []

    for t_cnt in range(task_num):
        ds_train = CityScapeDatasetWithinCitySplit(data_root=data_root, label_root=label_root, 
                                    transform=transform, split='train', city_names=city_splits[t_cnt], seed=seed)
        
        ds_test = CityScapeDatasetWithinCitySplit(data_root=data_root, label_root=label_root, transform=transform, 
                                   split='test', city_names=city_splits[t_cnt], seed=seed)
        
        train_datasets.append(ds_train)
        test_datasets.append(ds_test)

    
    for t_cnt in range(task_num):
        dl_train = DataLoader(train_datasets[t_cnt], batch_size=bs, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_ = []

        for epoch in range(nepoch):
            for i, data in enumerate(dl_train):
                image, mask = data
                image, mask = image.to(device), mask.squeeze().to(device)           

                optimizer.zero_grad()
                output = model(image)
                if multihead:
                    output = output[t_cnt]

                loss_critic = criterion(output, mask)
                if t_cnt > 0:    
                    loss_ewc = compute_ewc_loss(model)
                else:
                    loss_ewc = 0

                loss = loss_critic + loss_ewc * weight_ewc

                loss.backward()
                optimizer.step()

                loss_.append(loss.item())

                if (i+1) % eval_every == 0 or (i+1) == len(dl_train):
                    __ = eval(model, test_datasets[t_cnt], verbose=True, 
                              multihead=multihead, t_cnt=t_cnt)
                    
                    torch.save(model.state_dict(), f'{save_path}/{save_name}_model_t{t_cnt}.pth')

                if (i+1) % save_every == 0 or (i+1) == len(dl_train):
                    print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
                    
                    plt.plot(loss_)
                    plt.savefig(f'{save_path}/{save_name}_training_loss.png')
                    plt.close()

                    fig, ax = plt.subplots(3, 5, figsize=(20, 10))
                    #first row: input images second row: ground truth third row: predicted
                    #title accordingly
                    for i in range(5):
                        ax[0, i].imshow(image[i].cpu().detach().permute(1, 2, 0))
                        ax[0, i].set_title('input')

                        ax[1, i].imshow(mask[i].cpu().detach(), cmap='viridis', vmin=0, vmax=33)
                        ax[1, i].set_title('ground truth')

                        ax[2, i].imshow(output[i].cpu().detach().argmax(dim=0), cmap='viridis', vmin=0, vmax=33)
                        ax[2, i].set_title('predicted')

                    plt.savefig(f'{save_path}/{save_name}_training_results.png')
                    plt.close()



        register_ewc_params(model, dl_train, loss_type=loss_type, multihead=multihead, t_cnt=t_cnt)
        
        for eval_cnt in range(task_num):            
            acc_mat[t_cnt, eval_cnt]  = eval(model, test_datasets[eval_cnt], 
                                             verbose=False, multihead=multihead, t_cnt=t_cnt)

        np.save(f'{save_path}/{save_name}_acc_mat.npy', acc_mat)
        
        with np.printoptions(precision=2):
            print(acc_mat)

    print(f'Task splits are {city_splits}')



# train_one_task('leftImg8bit', 'clean_all_cities_FSunet')

# train_one_task('leftImg8bit', 'clean_all_cities_VGG19') 
# train_one_task('leftImg8bit', 'clean_all_cities_VGG19_focal_loss', loss='focal') 

# train_cont_cities_vanilla('leftImg8bit', 'clean_all_cities_VGG19_focalloss_cont_vanilla_t6_3pt', loss='focal')

# train_cont_cities_ewc('leftImg8bit', 'clean_all_cities_VGG19_focalloss_ewc_t6_1pt', loss_type='focal')

# train_cont_cities_ewc('leftImg8bit', 'clean_all_cities_VGG19_focalloss_ewc_t6_1pt_multihead', 
#                       loss_type='focal', multihead=True)

# train_cont_weathers_vanilla('weathers_VGG19_focalloss_cont_vanilla_t3', loss='focal')

