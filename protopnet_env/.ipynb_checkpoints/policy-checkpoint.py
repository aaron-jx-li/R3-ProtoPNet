from ray.rllib.policy.policy import Policy, PolicyState
"""
policy.py
    - Description: This file contains the code for the policy class. The computation of the bounding box information is conducted here. 
    
    - General Idea for 
        # Run k images from env through model
        
        # get k bounding boxes
        ## Look at how bounding boxes are computed in global_analysis.py/local_analysis.py for info
        
        # Pass images with bounding boxes to environment
        ## Or just the bounding box itself--need to make a design choice here 
        
"""

class PPNetPolicy(Policy):
    def __init__(self, model):
        self.PPnet = PPnet
    
    
    # Fill in
    def compute_action(self, obs_batch, prototype_network_parallel):
        heaps = []
        for _ in range(n_prototypes):
        # a heap in python is just a maintained list
            heaps.append([])
        proto_inputs = self.PPnet.conv_features(obs_batch)
        protoL_input_torch, proto_dist_torch = \
                prototype_network_parallel.module.push_forward(search_batch)
        for img_idx, distance_map in enumerate(proto_dist_):
            for j in range(n_prototypes):
                # find the closest patches in this batch to prototype j

                closest_patch_distance_to_prototype_j = np.amin(distance_map[j])


                closest_patch_indices_in_distance_map_j = \
                    list(np.unravel_index(np.argmin(distance_map[j],axis=None),
                                          distance_map[j].shape))
                closest_patch_indices_in_distance_map_j = [0] + closest_patch_indices_in_distance_map_j
                closest_patch_indices_in_img = \
                    compute_rf_prototype(search_batch.size(2),
                                         closest_patch_indices_in_distance_map_j,
                                         protoL_rf_info)
                closest_patch = \
                    search_batch_input[img_idx, :,
                                       closest_patch_indices_in_img[1]:closest_patch_indices_in_img[2],
                                       closest_patch_indices_in_img[3]:closest_patch_indices_in_img[4]]
                closest_patch = closest_patch.numpy()
                closest_patch = np.transpose(closest_patch, (1, 2, 0))

                original_img = search_batch_input[img_idx].numpy()
                original_img = np.transpose(original_img, (1, 2, 0))

                if prototype_network_parallel.module.prototype_activation_function == 'log':
                    act_pattern = np.log((distance_map[j] + 1)/(distance_map[j] + prototype_network_parallel.module.epsilon))
                elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                    act_pattern = max_dist - distance_map[j]
                else:
                    act_pattern = prototype_activation_function_in_numpy(distance_map[j])

                # 4 numbers: height_start, height_end, width_start, width_end
                patch_indices = closest_patch_indices_in_img[1:5]

                # construct the closest patch object
                closest_patch = ImagePatch(patch=closest_patch,
                                           label=search_y[img_idx],
                                           distance=closest_patch_distance_to_prototype_j,
                                           original_img=original_img,
                                           act_pattern=act_pattern,
                                           patch_indices=patch_indices)
                else:
                    closest_patch = ImagePatchInfo(label=search_y[img_idx],
                                                   distance=closest_patch_distance_to_prototype_j)
                # add to the j-th heap 
                if len(heaps[j]) < k:
                    heapq.heappush(heaps[j], closest_patch)
                else:
                    # heappushpop runs more efficiently than heappush
                    # followed by heappop
                    heapq.heappushpop(heaps[j], closest_patch)
        
        action = torch.cat(top_k_imgs, axis=1)
        
        return action
    
    # Update the plicy network (PPnet) given the reward (?)
    def learn_on_batch(self, batch):
        
        return
    
    
    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]