import torch
import sys
import importlib
from utils.sde_models.TC import TC
from utils.sde_models.model import base_Model
import torch.nn.functional as F
def model_load(model, temporal_contr_model, x, device, config,testuser):
    model.eval()
    temporal_contr_model.eval()
    path = testuser['sde']
    chkpoint = torch.load(path, map_location=device)
    model_dict= chkpoint["model_state_dict"]
    tc_dict = chkpoint['temporal_contr_model_state_dict']
    model.load_state_dict(model_dict)
    temporal_contr_model.load_state_dict(tc_dict)
    with torch.no_grad():
        x = x.float().to(device)       
        predictions1, features1 = model(x)
        features1 = F.normalize(features1, dim=1)
        try:
            c_t, s_t= temporal_contr_model.context(features1)
        except:
            c_t= temporal_contr_model.context(features1)
            s_t = c_t
    
    return c_t, s_t
def conditioner(x, y, testuser):
    device = 'cuda'
    selected_dataset = testuser['name'].split('_task')[0]
    module_name = f'config_files.{selected_dataset}_FEA_Configs'
    ConfigModule = importlib.import_module(module_name)
    configs = ConfigModule.Config()
    configs.TC.train_test = 1

    if x.shape[2] == 1:
        x = x.squeeze(2)
    configs.final_out_channels = x.shape[-1]
    configs.batch_size = x.shape[0]

    model = base_Model(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)
    c_t = model_load(model, temporal_contr_model, x, device, configs, testuser)
    return c_t
