import sys
sys.path.append("..")

from torch.optim.lr_scheduler import StepLR

import copy
import time
from time import gmtime, strftime
from datetime import datetime

from dynamicplot import DynamicPlot
from dataloader_graph import data_input_to_gmn
from dataloader_triplet import RICO_TripletDataset
from test_dataloader_triplet import test_RICO_TripletDataset
from combine_all_modules_6 import compute_similarity, reshape_and_split_tensor, make_graph_matching_net # this imports the util file

from cross_graph_communication_5 import *

import wandb

from inference.test_triplet_dataset import TestTripletDataset
from inference.sg_dataset import SGDataset

#####################################################
########### Some helper functions ###################
def set_lr2(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * decay_factor
    print('\n', optimizer, '\n')


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

################ End of helper functions ##############
#######################################################

def _main(config):
    
    if config.cuda and torch.cuda.is_available():
        print('Using CUDA on GPU', config.gpu)
    else:
        print('Not using CUDA')

    device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else "cpu")
    print(device)

    print('Initializing the model..........')

    wandb.init(project="layout_gmn", name="layoutgmn", tags=["v0.92"], config=config)

    if config.hardmining:
        wandb.run.tags = wandb.run.tags + ("hardmining",)
    
    if config.pretrained_wandb_model_ref is not None:
        wandb.run.tags = wandb.run.tags + ("pretrained",)

    assert wandb.run is not None

    model_save_path = f"{config.model_save_path}/{wandb.run.id}/"

    gmn_net = make_graph_matching_net(config)

    if config.pretrained_wandb_model_ref is None:
        print('No pretrained models loaded')
        gmn_model = gmn_net

        starting_epoch = 0

    else:

        pretrained_path, starting_epoch = download_model_weights_from_wandb(config.pretrained_wandb_model_ref)
        
        # print(model_save_path + 'gmn_tmp_model' + stored_epoch + '.pkl')
        

        gmn_model = load_pretrained_model(gmn_net, pretrained_path)
        
        print('Finished loading saved models')

    '''
    if torch.cuda.device_count() > 1 and config.cuda:
        print('Using', torch.cuda.device_count(), 'GPUs!')
        gmn_model = nn.DataParallel(gmn_model, device_ids=[0,1])  # , output_device=device)
    '''

    #gmn_model.to(f'cuda:{gmn_model.device_ids[0]}')
    gmn_model.to(device)
    #gmn_model.cuda()

    gmn_model_params = list(gmn_model.parameters())
    optimizer = torch.optim.Adam(gmn_model_params, lr=config.lr)
    #scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    header = '  Time      Epoch   Iter    GVS    s_pos   sm_neg   s_diff     Loss'
    log_template = ' '.join('{:>9s},{:>4.0f}/{:<4.0f},{:<5.0f},{:>6.4f},{:>7.5f},{:>7.5f},{:>7.5f}, {:>10.7f}'.split(','))

    iteration = 0
    epoch = starting_epoch

    total_samples = 0

    gmn_model.train()
    torch.set_grad_enabled(True)
    start = time.time()
    loader = RICO_TripletDataset(config)

    

    if config.test_triplets_csv is not None:
        sg_dataset = SGDataset(config.use_box_feats)

        test_triplets_dataset = TestTripletDataset(sg_dataset, config.test_triplets_csv)
    else:
        test_triplets_dataset = None

    while True:
        data =  loader.get_batch('train')#.to(device)

        sg_data_a = data['sg_data_a']
        sg_data_p = data['sg_data_p']
        sg_data_n = data['sg_data_n']

        GraphData = data_input_to_gmn(config, device, sg_data_a, sg_data_p, sg_data_n).quadruples()
        #GraphData = data_input_to_gmn(sg_data_a, sg_data_p, sg_data_n).quadruples()

        optimizer.zero_grad()

        graph_vectors = gmn_model(**GraphData)#.cuda()
        # print(graph_vectors)
        x1, y, x2, z = reshape_and_split_tensor(graph_vectors, 4)

        loss = triplet_loss(x1, y, x2, z, loss_type=config.loss_type, margin=config.margin_val)
        
        graph_vec_scale = torch.mean(graph_vectors ** 2)

        if config.graph_vec_regularizer_weight > 0:
            loss = loss + config.graph_vec_regularizer_weight * 0.5 * graph_vec_scale

        total_batch_loss = loss.mean()

        # Rescale to original loss scale (batch_size=20 was used in the original code):
        total_batch_loss = total_batch_loss * 20

        # Compute train triplet accuracy and similarities:
        with torch.no_grad():
            sim_positives = compute_similarity(config, x1, y)
            sim_negatives = compute_similarity(config, x2, z)

            sim_pos = torch.mean(sim_positives)
            sim_neg = torch.mean(sim_negatives)
            sim_diff = sim_pos - sim_neg

            train_triplet_accuracy = (sim_positives > sim_negatives).sum() / len(sim_positives)
        
        total_batch_loss.backward()
        clip_gradient(optimizer, config.clip_val)
        optimizer.step()
        torch.cuda.empty_cache()
        iteration += 1

        total_samples += len(sg_data_a)

        if epoch == starting_epoch and iteration == 1:

            print("Training Started ")
            print(header)

        epoch_done = data['bounds']['wrapped']

        if iteration % 1 == 0 or epoch_done:
            elsp_time = (time.time() - start)
            print(log_template.format(strftime(u"%H:%M:%S", time.gmtime(elsp_time)),
                                      epoch, config.epochs, iteration, graph_vec_scale,
                                      sim_pos, sim_neg, sim_diff, total_batch_loss.item()))

            wandb.log({
                "epoch": epoch,
                "epoch_iteration": iteration,
                "samples": total_samples,
                "graph_vec_scale": graph_vec_scale.cpu(),
                "sim_pos": sim_pos.cpu(),
                "sim_neg": sim_neg.cpu(),
                "sim_diff": sim_diff.cpu(),
                "train/triplet_accuracy": train_triplet_accuracy.cpu(),
                "total_batch_loss": total_batch_loss.item(),
                "elapsed_time": elsp_time,
            })

            '''
            with open(config.model_save_dir + '/log.txt', 'a') as f:
                f.write('Epoch [%02d] [%05d / %05d  ] Average_Loss: %.5f  Recon Loss: %.4f  DML Loss: %.4f\n' % (
                epoch + 1, iteration * opt.batch_size, len(loader), losses.avg, losses_recon.avg, losses_dml.avg))
                f.write('Completed {} images in {}'.format(iteration * opt.batch_size, elsp_time))
            '''
            #print('Completed {} images in {}'.format(iteration * config.batch_size, elsp_time))
            #start = time.time()


        if epoch_done:
            if test_triplets_dataset is not None:
                gmn_model.eval()
                with torch.no_grad():
                    test_triplet_accuracy = compute_test_triplet_accuracy(test_triplets_dataset, gmn_model, config, device)
                    
                gmn_model.train()

                print(f"{test_triplet_accuracy=}; {epoch=}; {total_samples=}")

                wandb.log({
                    "epoch": epoch,
                    "epoch_iteration": iteration,
                    "samples": total_samples,
                    "test/triplet_accuracy": test_triplet_accuracy.cpu(),
                })

            epoch += 1
            iteration = 0


        if epoch % config.save_network_every == 0 and epoch_done:
            os.makedirs(model_save_path, exist_ok=True)
            # os.makedirs(config.feature_save_path, exist_ok=True)
            try:

                checkpoint_epochs = epoch
                
                tmp_model_path = model_save_path + 'gmn_tmp_model' + str(checkpoint_epochs) + '.pkl'
                torch.save(gmn_model.state_dict(), tmp_model_path)

            except:
                print('failed to save temp models')
                raise
            
            model_art = wandb.Artifact(f"gmn_model_{wandb.run.id}", type="model", metadata={"epoch": checkpoint_epochs, "seen_samples": total_samples})
            model_art.add_file(tmp_model_path)
            wandb.run.log_artifact(model_art)

            # set model to training mode again after it had been in eval mode
            #gmn_model.train()
            #torch.set_grad_enabled(True)


        if epoch > 700:
            break


def compute_test_triplet_accuracy(test_triplets_dataset: TestTripletDataset, gmn_model, config, device):
    all_triplets = test_triplets_dataset.get_batch_of_all_triplets()

    graph_data_dict = data_input_to_gmn(config, device,
                                        all_triplets['sg_data_a'], all_triplets['sg_data_p'], all_triplets['sg_data_n']).quadruples()
    
    with torch.no_grad():

        graph_vecs = gmn_model(**graph_data_dict)
        x1, y, x2, z = reshape_and_split_tensor(graph_vecs, 4)

        sim_poss = compute_similarity(config, x1, y) # these are now list of tensors \n;
        # previously there was torch.mean at the start of the RHS
        sim_negs = compute_similarity(config, x2, z) # list of tensors
    
    triplet_accuracy = (sim_poss > sim_negs).sum() / len(sim_poss)

    return triplet_accuracy


def download_model_weights_from_wandb(pretrained_wandb_model_ref):

    assert wandb.run is not None, "wandb.run is None; wandb.init() must be called before download_model_weights_from_wandb()"

    model_art: wandb.Artifact = wandb.run.use_artifact(pretrained_wandb_model_ref, type="model")

    pretrained_path = model_art.file()

    starting_epoch = min(int(model_art.metadata["epoch"]), int(model_art.logged_by().summary["epoch"]))

    if starting_epoch < model_art.metadata["epoch"]:
        print(f"WARNING: Loaded model has incorrect epoch metadata: {starting_epoch=} < {model_art.metadata['epoch']=}")
        print("Using max epoch of run as starting epoch")

    print(f'Downloaded pretrained model to:  {pretrained_path} (wandb reference: {pretrained_wandb_model_ref})')

    return pretrained_path, starting_epoch
    

def load_pretrained_model(gmn_model, pretrained_path):
    '''
    :param gmn_model: network model
    :param save_dir: path of the dir where the models have been saved
    :param stored_epoch: str, ex: '8'
    '''
    print('Loading pretrained models')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gmn_model_state_dict = torch.load(pretrained_path, map_location=torch.device(device))

    from collections import OrderedDict

    def remove_module_fromStateDict(model_state_dict, model):
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[0:]  # if ran on two GPUs, remove 'module.module.'; else, no change
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model

    loaded_gmn_model = remove_module_fromStateDict(gmn_model_state_dict, gmn_model)
    print('Finished loading checkpoint')
    return loaded_gmn_model


#os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
if __name__ == '__main__':
    config = get_args()

    assert config.train_mode != config.eval_mode
    if config.train_mode and not config.eval_mode:
        _main(config)
    else:
        raise NotImplementedError(f"Expected train mode, but {config.train_mode=}, {config.eval_mode=}")

