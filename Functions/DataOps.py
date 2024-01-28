import torch

# Def feature maps -> gram matrices function
def feature_to_gram(feature_maps : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    feature_maps_flattened = { key : torch.flatten(value, start_dim=1) for key, value in feature_maps.items() } 
    Gram_Matrices = { key : torch.matmul(value, torch.transpose(value, 0, 1)) for key, value in feature_maps_flattened.items() }
    return Gram_Matrices


# Def Loss Function （总体loss）
def loss(feature_maps       : dict[str, torch.Tensor], 
         train_feature_maps : dict[str, torch.Tensor],
         weights            : dict[str, float],
         device             : torch.cuda.device
        ) -> torch.Tensor:
    
    # Define each gram matrix loss （对某一层的一对gram matrix算loss）
    def each_loss(gram_matrix : torch.Tensor, train_gram_matrix : torch.Tensor, shape : tuple[int]) -> torch.Tensor:
        assert train_gram_matrix.requires_grad == True
        assert gram_matrix.requires_grad       == False
        assert gram_matrix.shape == train_gram_matrix.shape
        tmp = gram_matrix - train_gram_matrix
        res = torch.pow(torch.linalg.matrix_norm(tmp), 2) / (4 * shape[0] * shape[0] * shape[1] * shape[1])
        return res
        
        
    shapes = { key : item.shape for key, item in feature_maps.items() }
    
    # 得到总体的gram matrix
    gram_matrices       = feature_to_gram(feature_maps)
    train_gram_matrices = feature_to_gram(train_feature_maps)
    
    # 初始化返回值
    ret_value = torch.tensor([0.], requires_grad=False).to(device)
    
    assert len(gram_matrices) == len(train_gram_matrices)
    
    # 计算总loss
    for layer_name, weight in weights.items():
        ret_value += weight * each_loss(gram_matrix=gram_matrices[layer_name],
                                        train_gram_matrix=train_gram_matrices[layer_name],
                                        shape=shapes[layer_name])
    return ret_value