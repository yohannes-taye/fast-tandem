/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/
#include<iostream>
#include <thread>

#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>

#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "dr_mvsnet.h"
#include "util/commandline.h"
#include "util/DatasetReader.h"
#include "util/Timer.h"
#include "IOWrapper/ImageRW.h"

using namespace std; 
using namespace dso; 
using namespace cv; 

// int main(int argc, char **argv) {
//     cout<<"......WE OUT HERE HUSTLING...."<<endl;

//     DrMvsnet *mvsnet;
//     CommandLineOptions opt;
//     DrMvsnetOutput *output_previous;
//     tandemDefaultSettings(opt, argv[1]);
//     for (int i = 2; i < argc; i++) parseArgument(opt, argv[i]);
//     printSettings(opt);


//     mvsnet = new DrMvsnet("/home/tmc/tandem/tandem/exported/tandem/model.pt");
//     if (!test_dr_mvsnet(*mvsnet, "/home/tmc/tandem/tandem/exported/tandem/sample_inputs.pt", true, 4)) {
//       printf("Couldn't load MVSNet successfully.");
// //                exit(EXIT_FAILURE);
//     }

//     ImageFolderReader *reader;
//     if (!rgbd_flag)
//         reader = new ImageFolderReader(opt.source, opt.calib, opt.gammaCalib, opt.vignette);
//     else
//         reader = new RGBDReader(opt.source, opt.rgbdepth_folder, opt.calib, opt.gammaCalib, opt.vignette);
//     reader->setGlobalCalibration();
    
//     if (setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0) {
//         printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
//         exit(1);
//     }
//     int lstart = opt.start;
//     int lend = opt.end;
//     int linc = 1;
//     if (opt.reverse) {
//         printf("REVERSE!!!!");
//         lstart = opt.end - 1;
//         if (lstart >= reader->getNumImages())
//             lstart = reader->getNumImages() - 1;
//         lend = opt.start;
//         linc = -1;
//     }    
//     std::thread runthread([&]() {
//         std::vector<int> idsToPlay;
//         std::vector<double> timesToPlayAt;
//         for (int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i += linc) {
//             cout<<"Running "<<i<<endl;
//             idsToPlay.push_back(i);
//             if (timesToPlayAt.size() == 0) {
//                 timesToPlayAt.push_back((double) 0);
//             } else {
//                 double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
//                 double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
//                 timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / opt.playbackSpeed);
//             }
//         }
//         std::vector<ImageAndExposure *> preloadedImages;
//         std::vector<unsigned char *> preloadedImagesBGR;
//         if (opt.preload) {
//             printf("LOADING ALL IMAGES!\n");
//             for (int ii = 0; ii < (int) idsToPlay.size(); ii++) {
//                 int i = idsToPlay[ii];
//                 preloadedImages.push_back(reader->getImage(i));
//                 preloadedImagesBGR.push_back(reader->getImageBGR_8UC3_undis(i, reader->undistort->get_remapX(), reader->undistort->get_remapY()));
//             }
//         }
//         struct timeval tv_start;
//         gettimeofday(&tv_start, NULL);
//         clock_t started = clock();
//         const auto timer_start = Timer::start();
//         double sInitializerOffset = 0;


//         for (int ii = 0; ii < (int) idsToPlay.size(); ii++) {
//             int i = idsToPlay[ii];
//             ImageAndExposure *img;
//             RGBDepth *depth = nullptr;
//             dvo::core::RgbdImagePyramid *dvo_img = nullptr;
//             unsigned char *img_bgr = nullptr;
//             if (opt.preload) {
//                 img = preloadedImages[ii];
//                 img_bgr = preloadedImagesBGR[ii];
//             } else {
//                 img = reader->getImage(i); //                img_bgr = reader->getImageBGR_8UC3(i);
//                 img_bgr = reader->getImageBGR_8UC3_undis(i, reader->undistort->get_remapX(), reader->undistort->get_remapY());
                
//             }
            
//             output_previous = mvsnet->GetResult();
//             float const* image = output_previous->depth_dense; 
//             uint8_t greyArr[h][w];
//             for (int _h = 0; _h < h; _h++) {
//                 for (int _w = 0; _w < w; _w++) {
//                     const int i = _h*w + _w;
//                     const float valf = (image[i] - depth_min) / (depth_max - depth_min);
//                     const unsigned char val = (unsigned char) (255.0 *valf);
//                     greyArr[_h][_w] = val; 
//                 }
//             }
//             string greyArrWindow = "Depth Image";
//             namedWindow(greyArrWindow, WINDOW_AUTOSIZE);
//             Mat greyImg = Mat(h, w, CV_8U, &);
//             imshow(greyArrWindow, output_previous->depth);
//             waitKey(1);
//             delete img;
//             if (rgbd_flag) {
//                 delete depth;
//             }
//         }
//     }); 
//     runthread.join();
//     return 0; 
// }
vector<Mat> images;
Undistort *undistorter;
bool test_dr_mvsnet2(DrMvsnet &model, char const *filename_inputs, bool print, int repetitions, char const *out_folder);
int main(int argc, char **argv) {
    using std::cout;
    using std::endl;

    std::string module_path;
    std::string sample_path;
    int repetitions = 1;
    char const* out_folder = NULL;
    char const* image_folder = NULL; 
    char const* image_calibration_file = NULL;
    out_folder = "/home/tmc/tandem/tandem/src/output/";


    CommandLineOptions opt;
    tandemDefaultSettings(opt, argv[1]);
    for (int i = 2; i < argc; i++) parseArgument(opt, argv[i]);
        printSettings(opt);

    // ImageFolderReader *reader;
    // ImageAndExposure *img; 
    // unsigned char *img_bgr = nullptr;
    // reader = new ImageFolderReader(opt.source, opt.calib, opt.gammaCalib, opt.vignette);
    // cout<<"Number of images: "<<reader->getNumImages()<<endl; 
    vector<cv::String> fn;


    glob("/home/tmc/euroc_tandem_format_1.1.beta/euroc_tandem_format/V1_01_easy/images/*.png", fn, false);
    undistorter = Undistort::getUndistorterForFile(opt.calib, opt.gammaCalib, opt.vignette);
    setGlobalCalib((int) undistorter->getSize()[0], (int) undistorter->getSize()[1], undistorter->getK().cast<float>());
    //reader->setGlobalCalibration();
    
    size_t count = fn.size(); //number of png files in images folder
    string greyArrWindow = "Depth Image";
    cv::namedWindow(greyArrWindow, cv::WINDOW_AUTOSIZE);
    for (size_t i=0; i<count; i++){
        cv::Mat x = imread(fn[i]);
        unsigned char *bgr = x.data;

        cv::Mat img_gray;
        cv::cvtColor(x, img_gray, cv::COLOR_BGR2GRAY);
        ImageAndExposure img_dso(undistorter->w, undistorter->h);
        cv::Mat img_dso_wrapper(undistorter->h, undistorter->w, CV_32F, img_dso.image);
        img_gray.convertTo(img_dso_wrapper, CV_32F, 1.0);
        // model.CallAsync(
        //         height,
        //         width,
        //         view_num,
        //         ref_index,
        //         bgrs,
        //         intrinsic_matrix,
        //         c2ws,
        //         depth_min,
        //         depth_max,
        //         discard_percentage
        //     );


        // cv::cvtColor(x, x, RGB2BGR);
        // cv::imshow("RGB", frame);
        // cv::waitKey(1);
        // images.push_back(x);
    }

// for (size_t i=0; i<count; i++){
//         std::string fileName = fn[i].operator std::string(); 
//         MinimalImageB* vm8 = IOWrap::readImageBW_8U(fileName);
//  cv::cvtColor(frame, frame, BGR2RGB);
//         cv::Mat x = imread(fn[i]);
//         // images.push_back(x);
//     }

    // for (int i = 1 ; i < reader->getNumImages(); i += 1){
    //     img = reader->getImage(i); 
    //     float* x = img->image;
    //     cout<<"Iterating: "<<sizeof(x)/sizeof(float)<<endl;
    //     cout<<"Width: "<<img->w<<endl; 
    //     cout<<"Height: "<<img->h<<endl; 
    //     cout<<"Array size: "<<img->w * img->h<<endl; 
    //     cout<<"soemthing: "<<x[(img->h * img->w) - 1]<<endl;
    //     img_bgr = reader->getImageBGR_8UC3_undis(i, reader->undistort->get_remapX(), reader->undistort->get_remapY());
        
    //     cv::Mat image = cv::Mat(img->w, img->h, CV_8UC3, img->image); 
    //     string greyArrWindow = "Depth Image";

    //     cv::namedWindow(greyArrWindow, cv::WINDOW_AUTOSIZE);
    //     cv::imshow(greyArrWindow, image);
    //     waitKey(0);
    // }
    
    auto start = std::chrono::high_resolution_clock::now();
    DrMvsnet mvsnet("/home/tmc/tandem/tandem/exported/tandem/model.pt");
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
    cout << "Loading Model: " << (double) elapsed / 1000000.0 << " s" << endl;

    bool correct = test_dr_mvsnet2(mvsnet, "/home/tmc/tandem/tandem/exported/tandem/sample_inputs.pt", true, repetitions, out_folder);

    if (!correct) {
        return -1;
    }

    return 0;
}

bool test_dr_mvsnet2(DrMvsnet &model, char const *filename_inputs, bool print, int repetitions, char const *out_folder) {
  using std::cerr;
  using std::endl;
  using std::cout;

  constexpr int batch_size = 1;
  constexpr int channels = 3;

  using torch::kCPU;
  /* --- Convert Tensors to C data -- */

  /* ---  Load Input ---*/
  torch::jit::script::Module tensors = torch::jit::load(filename_inputs);

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;

  // image: (B, V, C, H, W)
  auto image = tensors.attr("image").toTensor().to(kCPU);
  if (image.size(0) != batch_size) {
    cerr << "Incorrect batch size." << endl;
    return false;
  }
  if (image.size(2) != channels) {
    cerr << "Incorrect channels." << endl;
    return false;
  }

  const int view_num = image.size(1);
  const int ref_index = view_num - 2;
  print = 1; 
  if (print)
    cout << "View Num: " << view_num << ", ref index: " << ref_index << endl;

  const int height = image.size(3);
  const int width = image.size(4);

  unsigned char *bgrs[view_num];
  for (int view = 0; view < view_num; view++)
    bgrs[view] = (unsigned char *) malloc(sizeof(unsigned char) * height * width * channels);

  auto image_a = image.accessor<float, 5>();
  for (int view = 0; view < view_num; view++) {
    unsigned char *bgr = bgrs[view];
    for (int h = 0; h < height; h++)
      for (int w = 0; w < width; w++) {
        // RGB -> BGR
        bgr[channels * (width * h + w) + 0] = (unsigned char) (255.0 * image_a[0][view][2][h][w]);
        bgr[channels * (width * h + w) + 1] = (unsigned char) (255.0 * image_a[0][view][1][h][w]);
        bgr[channels * (width * h + w) + 2] = (unsigned char) (255.0 * image_a[0][view][0][h][w]);
      }
  }

  auto intrinsic_matrix_tensor = tensors.attr("intrinsic_matrix.stage3").toTensor().to(kCPU);
  auto intrinsic_matrix_tensor_a = intrinsic_matrix_tensor.accessor<float, 3>();
  float *intrinsic_matrix = (float *) malloc(sizeof(float) * 3 * 3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      intrinsic_matrix[i * 3 + j] = intrinsic_matrix_tensor_a[0][i][j];

  // cam_to_world: (B, V, 4, 4)
  auto c2w_tensor = tensors.attr("cam_to_world").toTensor().to(kCPU);
  auto c2w_tensor_a = c2w_tensor.accessor<float, 4>();
  float **c2ws = (float **) malloc(sizeof(float *) * view_num);
  for (int view = 0; view < view_num; view++) {
    c2ws[view] = (float *) malloc(sizeof(float) * 4 * 4);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        c2ws[view][i * 4 + j] = c2w_tensor_a[0][view][i][j];
  }

  float depth_min, depth_max, discard_percentage;

  auto depth_min_tensor = tensors.attr("depth_min").toTensor().to(kCPU);
  auto depth_max_tensor = tensors.attr("depth_max").toTensor().to(kCPU);
  depth_min = depth_min_tensor.accessor<float, 1>()[0];
  depth_max = depth_max_tensor.accessor<float, 1>()[0];

  auto discard_percentage_tensor = tensors.attr("discard_percentage").toTensor().to(kCPU);
  discard_percentage = discard_percentage_tensor.accessor<float, 1>()[0];

  constexpr int stage = 3;
  auto depth_ref = tensors.attr("outputs.stage" + std::to_string(stage) + ".depth").toTensor().to(kCPU);
  auto confidence_ref = tensors.attr(
      "outputs.stage" + std::to_string(stage) + ".confidence").toTensor().to(kCPU);

  double elapsed1 = 0.0;
  double elapsed2 = 0.0;
  double elapsed3 = 0.0;

  bool correct = true;

  int warmup = (repetitions == 1) ? 0 : 5;

  for (int rep = 0; rep < repetitions + warmup; rep++) {
    if (rep == warmup) {
      elapsed1 = 0.0;
      elapsed2 = 0.0;
      elapsed3 = 0.0;
    }
    auto start = std::chrono::high_resolution_clock::now();
    //TODOME: I stoped looking for how theintrinsic_matrix was defined 
    model.CallAsync(
        height,
        width,
        view_num,
        ref_index,
        bgrs,
        intrinsic_matrix,
        c2ws,
        depth_min,
        depth_max,
        discard_percentage
    );
    elapsed1 += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    start = std::chrono::high_resolution_clock::now();
    bool ready = model.Ready();
    elapsed2 += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    if (print && ready)
      std::cout << "Was ready directly. Quite unexpected. Debug. " << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto output = model.GetResult();
    elapsed3 += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    auto depth_out = torch::from_blob(output->depth, {height, width});
    auto confidence_out = torch::from_blob(output->confidence, {height, width});

    auto error_depth = torch::mean(torch::abs(depth_out - depth_ref)).item().toFloat();
    auto error_confidence = torch::mean(torch::abs(confidence_out - confidence_ref)).item().toFloat();

    const double atol = 1e-2;
    auto correct_depth = error_depth < atol;
    auto correct_confidence = error_confidence < atol;
    if (print) {
      cout << "Correctness:" << endl;
      cout << "\tDepth correct     : " << correct_depth << ", error: " << error_depth << endl;
      cout << "\tConfidence correct: " << correct_confidence << ", error: " << error_confidence << endl;
    }

    correct &= correct_depth;
    correct &= correct_confidence;

    if (out_folder && rep == 0) {
      std::string out_name = std::string(out_folder) + "pred_outputs.pt";
      cout << "Writing Result to: " << out_name << endl;

      //      Vesion 1.6
      //      torch::save(out_name, "x.pt"); // this is actually a zip file

      //      Version 1.5
      auto bytes = torch::jit::pickle_save(depth_out);
      std::ofstream fout(out_name, std::ios::out | std::ios::binary);
      fout.write(bytes.data(), bytes.size());
      fout.close();
      cout<<"WHAT THE FUCK!"<<endl;
    }

    delete output;
  }

  if (print) {
    cout << "Performance:" << endl;
    cout << "\tCallAsync     : " << (double) elapsed1 / (1000.0 * repetitions) << " ms" << endl;
    cout << "\tReady         : " << (double) elapsed2 / (1000.0 * repetitions) << " ms" << endl;
    cout << "\tGetResult     : " << (double) elapsed3 / (1000.0 * repetitions) << " ms" << endl;
  }

  if (correct) {
    if (print)
      cout << "All looks good!" << endl;
    return true;
  } else {
    if (print)
      cout << "There has been an error. Do not use the model." << endl;
    return false;
  }
}
