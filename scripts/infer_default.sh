python -m main \
    +name=single_image_to_short \
    dataset=realestate10k_mini \
    algorithm=dfot_video_pose \
    experiment=video_generation \
    @diffusion/continuous \
    load=pretrained:DFoT_RE10K.ckpt \
    'experiment.tasks=[validation]' \
    experiment.validation.data.shuffle=True \
    dataset.context_length=1 \
    dataset.frame_skip=20 \
    dataset.n_frames=8 \
    experiment.validation.batch_size=1 \
    algorithm.tasks.prediction.history_guidance.name=vanilla \
    +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0


# python -m main \
#     +name=single_image_to_short \
#     dataset=realestate10k_mini \
#     algorithm=dfot_video_pose \
#     experiment=video_generation \
#     @diffusion/continuous \
#     load=pretrained:DFoT_RE10K.ckpt \
#     'experiment.tasks=[validation]' \
#     experiment.validation.data.shuffle=True \
#     dataset.max_frames=4 \
#     dataset.context_length=1 \
#     dataset.frame_skip=40 \
#     dataset.n_frames=4 \
#     experiment.validation.batch_size=1 \
#     algorithm.tasks.prediction.history_guidance.name=vanilla \
#     +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0
