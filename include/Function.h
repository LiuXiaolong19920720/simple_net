#include<opencv2\core\core.hpp>
#include<iostream>

namespace liu
{

	//sigmoid function
	cv::Mat sigmoid(cv::Mat &x);

	//Tanh function
	cv::Mat tanh(cv::Mat &x);

	//ReLU function
	cv::Mat ReLU(cv::Mat &x);

	//Derivative function
	cv::Mat derivativeFunction(cv::Mat& fx, std::string func_type);

	//Objective function
	void calcLoss(cv::Mat &output, cv::Mat &target, cv::Mat &output_error, float &loss);


}