//
// Created by siky on 2022/2/19.
//
#include <iostream>
#include "AudioFile.h"
#include "unsupported/Eigen/FFT"
#include "Eigen/Eigen"
#include "Eigen/Core"
#include "onnxruntime_cxx_api.h"
#include "CsvPraser.h"
typedef Eigen::Matrix<float,1,Eigen::Dynamic,Eigen::RowMajor> Matrix1Xf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorcf;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixf;

static Vectorf pad(Vectorf &x, int left, int right, const std::string &mode, float value){
    Vectorf x_paded = Vectorf::Constant(left+x.size()+right, value);
    x_paded.segment(left, x.size()) = x;

    if (mode.compare("reflect") == 0){
        for (int i = 0; i < left; ++i){
            x_paded[left-(i+1)] = x_paded[left+(i+1)];
        }
        for (int i = 0; i < right; ++i){
            x_paded[i+x.size()+left] = x_paded[x.size()-(i+1)+left-1];
        }
    }

    if (mode.compare("symmetric") == 0){
        for (int i = 0; i < left; ++i){
            x_paded[i] = x[left-i-1];
        }
        for (int i = left; i < left+right; ++i){
            x_paded[i+x.size()] = x[x.size()-1-i+left];
        }
    }

    if (mode.compare("edge") == 0){
        for (int i = 0; i < left; ++i){
            x_paded[i] = x[0];
        }
        for (int i = left; i < left+right; ++i){
            x_paded[i+x.size()] = x[x.size()-1];
        }
    }
    return x_paded;
}


static Vectorf mel_sepctrogram(Vectorf &data, int sr=44100, int n_fft=2048, int n_hop=1024,
                               const std::string &win = "hann", bool center = false,
                               const std::string &mode = "reflect", float power = 2.0f,
                               int n_mels = 128, int f_min=0, int f_max=44100/2)
{
    auto fft = Eigen::FFT<float>();
    Vectorf window = 0.5*(1.f-(Vectorf::LinSpaced(n_fft, 0.f, static_cast<float>(n_fft-1))*2.f*M_PIf64/n_fft).array().cos());
    Vectorf x_paded = pad(data, n_fft/2, n_fft/2, "reflect", 0.f);
    int n_f = n_fft/2+1; //n_fft_real
    int n_frames = 1+(x_paded.size()-n_fft) / n_hop;
    Matrixcf X(n_frames, n_fft);

    for (int i = 0; i < n_frames; ++i){
        Vectorf x_frame = window.array()*x_paded.segment(i*n_hop, n_fft).array();
        if(false && i%10==0) {
            std::cout << "fft_input:(" << i << "):";
            for (int j = 0; j < x_frame.size(); j++)
                std::cout << x_frame[j] << ",";
            std::cout << std::endl;
        }
        X.row(i) = fft.fwd(x_frame);
        if(false && i%10==0) {
            std::cout << "fft_output:(" << i << "):";
            for (int j = 0; j < X.row(i).size(); j+=100)
                std::cout << X.row(i)[j] << ",";
            std::cout << std::endl;
        }

    }
    Matrixcf stft_result = X.leftCols(n_f);
    Matrixf power_stft = stft_result.array().abs2();
    for (auto item : power_stft.col(0))
    {
        std::cout<<item<<",";
    }
    std::cout<<std::endl;

//    Matrixf mel_basis = melfilter(sr, n_fft, n_mels, 0, 20000);
//    Matrixf sp = spectrogram(stft_result, 2.0f);
//    Matrixf testsound_spec = mel_basis*sp.transpose();
//    std::cout << "mel_output:";
//    for(int i =0;i <testsound_spec.rows();i++) {
//        for (int j = 0; j < testsound_spec.cols(); j += 1)
//            std::cout << testsound_spec.row(i)[j] << ",";
//        std::cout << std::endl;
//    }
    return power_stft;
}

int main()
{

    Vectorf test_vector = Eigen::Map<Vectorf>(std::vector<float>{1,2,3,4,5,6,7,8,9}.data(),9);
    Vectorf result = pad(test_vector,7,7,"reflect",0);
    for(auto item: result)
    {
        std::cout <<item<<",";
    }
    std::cout<<std::endl;
    AudioFile<float> audioFile;
    audioFile.load("../barking_recorded.wav");
    auto samples = audioFile.samples;
    Vectorf data = Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>(samples.data()->data(),samples.size(),samples[0].size()).colwise().mean();

    auto r = mel_sepctrogram(data);

    return 0;
}