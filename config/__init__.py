# config = {
    #     "dataset" : "isaac_office_all_fisheye",
    #     # Dataset available are : paris6k, triplet, eng3_floor1_fisheye,
    #     # isaac_office_all_fisheye, isl2_3places_fisheye, isl2_3places_pinhole, isl2_places_fisheye
    #     "model_config": {
    #         "model_name" : "vprmodel",
    #         "feature_extractor" : {
    #             # Choices Are : rps, vgg16, resnet18
    #             "model" : "resnet18",
    #             "fine_tuning" : True,
    #         },
    #         "clustering" : {
    #             "model" : "netvlad",
    #             "num_clusters" : 64,
    #             "desc_dim" : 512,
    #             "normalize" : True,
    #             "normalize_input": True,
    #             "whiten" : True,
    #             "alpha": 100.0,             # 100.0
    #         },
    #     },
    #     "loss" : {
    #         "name" : "hardtripletloss",
    #         "margin" : 0.1,
    #         "hardest" : False,
    #         "squared" : False,
    #     },
    #     "training_epoch" : 100,
    #     "eval_config" : {

    #     },
    #     "enable_tensorboard" : True,
    # }