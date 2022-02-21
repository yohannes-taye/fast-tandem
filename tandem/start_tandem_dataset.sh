# /home/tmc/tandem/tandem/build/bin/tandem_dataset \
#       preset=gui \
#       result_folder=/home/tmc/tandem/tandem_result \
#       files=/home/tmc/tandem/camera_calibrate/output_scene1/resized \
#       calib=/home/tmc/tandem/camera_calibrate/output_scene1/camera2.txt \
#       mvsnet_folder=/home/tmc/tandem/tandem/exported/tandem \
#       mode=1


./tandem_dataset \
      preset=gui \
      result_folder=/home/tmc/Project/tandem_result \
      files=/home/tmc/euroc_tandem_format_1.1.beta/euroc_tandem_format/V1_01_easy/images \
      calib=/home/tmc/euroc_tandem_format_1.1.beta/euroc_tandem_format/V1_01_easy/camera.txt \
      mvsnet_folder=/home/tmc/tandem/tandem/exported/tandem \
      mode=1

# build/bin/tandem_dataset \
#       preset=gui \
#       result_folder=/home/tmc/Project/tandem_result \
#       files=/home/tmc/Project/euroc_tandem_dataset/euroc_tandem_format/V1_01_easy/images \
#       calib=/home/tmc/Project/euroc_tandem_dataset/euroc_tandem_format/V1_01_easy/camera1.txt \
#       mvsnet_folder=/home/tmc/Project/tandem/tandem/exported/tandem/model.pt \
#       mode=1


#FOR EUROC data set 
# build/bin/tandem_dataset \
#       preset=gui \
#       result_folder=/media/zhy/1484975B84973E64/lali/data/euroc_tandem_format/output/ \
#       files=/media/zhy/1484975B84973E64/lali/data/euroc_tandem_format/V1_01_easy/images \
#       calib=/media/zhy/1484975B84973E64/lali/data/euroc_tandem_format/V1_01_easy/camera.txt \
#       mvsnet_folder=/home/zhy/Projects/lali/tandem/tandem/exported/tandem \
#       mode=1


# build/bin/tandem_dataset \
#     preset= /media/zhy/1484975B84973E64/lali/data/euroc_tandem_format/V1_01_easy \
#     result_folder=/media/zhy/1484975B84973E64/lali/data/euroc_tandem_format/output \
#     files=/media/zhy/1484975B84973E64/lali/data/euroc_tandem_format/V1_01_easy/images \
#     calib=/media/zhy/1484975B84973E64/lali/data/euroc_tandem_format/V1_01_easy \
#     mvsnet_folder=/home/zhy/Projects/lali/tandem/cva_mvsnet/pretrained/ablation/abl01_baseline.pkl \
#     mode=1