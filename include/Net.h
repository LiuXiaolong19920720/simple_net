
#ifndef NET_H
#define NET_H

#endif // NET_H

#pragma once

#include <iostream>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
//#include<iomanip>

#include"Function.h"

namespace liu
{

	class Net
	{
	public:
        //Integer vector specifying the number of neurons in each layer including the input and output layers.
		std::vector<int> layer_neuron_num;
		std::string activation_function = "sigmoid";
		int output_interval = 10;
		float learning_rate; 
		float accuracy = 0.;
		std::vector<double> loss_vec;
		float fine_tune_factor = 1.01;

	protected:
		std::vector<cv::Mat> layer;
		std::vector<cv::Mat> weights;
		std::vector<cv::Mat> bias;
		std::vector<cv::Mat> delta_err;

		cv::Mat output_error;
		cv::Mat target;
		cv::Mat board;
		float loss;

	public:
		Net() {};
		~Net() {};

		//Initialize net:genetate weights matrices¡¢layer matrices and bias matrices
		// bias default all zero
		void initNet(std::vector<int> layer_neuron_num_);

		//Initialise the weights matrices.
		void initWeights(int type = 0, double a = 0., double b = 0.1);

		//Initialise the bias matrices.
		void initBias(cv::Scalar& bias);

		//Forward
		void forward();

		//Forward
		void backward();

		//Train,use accuracy_threshold
		void train(cv::Mat input, cv::Mat target, float accuracy_threshold);

		//Train,use loss_threshold
		void Net::train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve = false);

		//Test
		void test(cv::Mat &input, cv::Mat &target_);

		//Predict,just one sample
		int predict_one(cv::Mat &input);

		//Predict,more  than one samples
		std::vector<int> predict(cv::Mat &input);

		//Save model;
		void save(std::string filename);

		//Load model;
		void load(std::string filename);

	protected:
		//initialise the weight matrix.if type =0,Gaussian.else uniform.
		void initWeight(cv::Mat &dst, int type, double a, double b);

		//Activation function
		cv::Mat activationFunction(cv::Mat &x, std::string func_type);

		//Compute delta error
		void deltaError();

		//Update weights
		void updateWeights();
	};

	//Get sample_number samples in XML file,from the start column. 
	void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);

	// Draw loss curve
	void draw_curve(cv::Mat& board, std::vector<double> points);
}