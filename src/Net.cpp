//Implement the function included in net.h
#include"../include/Net.h"

namespace liu
{
	//Activation function
	cv::Mat Net::activationFunction(cv::Mat &x, std::string func_type)
	{
		activation_function = func_type;
		cv::Mat fx;
		if (func_type == "sigmoid")
		{
			fx = sigmoid(x);
		}
		if (func_type == "tanh")
		{
			fx = tanh(x);
		}
		if (func_type == "ReLU")
		{
			fx = ReLU(x);
		}
		return fx;
	}

	//Initialize net
	void Net::initNet(std::vector<int> layer_neuron_num_)
	{
		layer_neuron_num = layer_neuron_num_;
		//Generate every layer.
		layer.resize(layer_neuron_num.size());
		for (int i = 0; i < layer.size(); i++)
		{
			layer[i].create(layer_neuron_num[i], 1, CV_32FC1);
		}
		std::cout << "Generate layers, successfully!" << std::endl;

		//Generate every weights matrix and bias
		weights.resize(layer.size() - 1);
		bias.resize(layer.size() - 1);
		for (int i = 0; i < (layer.size() - 1); ++i)
		{
			weights[i].create(layer[i + 1].rows, layer[i].rows, CV_32FC1);
			//bias[i].create(layer[i + 1].rows, 1, CV_32FC1);
			bias[i] = cv::Mat::zeros(layer[i + 1].rows, 1, CV_32FC1);
		}
		std::cout << "Generate weights matrices and bias, successfully!" << std::endl;
		std::cout << "Initialise Net, done!" << std::endl;
	}

	//initialise the weights cv::Matrix.if type =0,Gaussian.else uniform.
	void Net::initWeight(cv::Mat &dst, int type, double a, double b)
	{
		if (type == 0)
		{
			randn(dst, a, b);
		}
		else
		{
			randu(dst, a, b);
		}
	}

	//initialise the weights matrix.
	void Net::initWeights(int type, double a, double b)
	{
		//Initialise weights matrices and bias
		for (int i = 0; i < weights.size(); ++i)
		{
			initWeight(weights[i], 0, 0., 0.1);
		}
	}

	//Initialise the bias matrices.
	void Net::initBias(cv::Scalar& bias_)
	{
		for (int i = 0; i < bias.size(); i++)
		{
			bias[i] = bias_;
		}
	}

	//Forward
	void Net::forward()
	{
		for (int i = 0; i < layer_neuron_num.size() - 1; ++i)
		{
			cv::Mat product = weights[i] * layer[i] + bias[i];
			layer[i + 1] = activationFunction(product, activation_function);
		}
		calcLoss(layer[layer.size() - 1], target, output_error, loss);
	}

	//Compute delta error
	void Net::deltaError()
	{
		delta_err.resize(layer.size() - 1);
		for (int i = delta_err.size() - 1; i >= 0; i--)
		{
			delta_err[i].create(layer[i + 1].size(), layer[i + 1].type());
			//cv::Mat dx = layer[i+1].mul(1 - layer[i+1]);
			cv::Mat dx = derivativeFunction(layer[i + 1], activation_function);
			//Output layer delta error
			if (i == delta_err.size() - 1)
			{
				delta_err[i] = dx.mul(output_error);
			}
			else  //Hidden layer delta error
			{
				cv::Mat weight = weights[i];
				cv::Mat weight_t = weights[i].t();
				cv::Mat delta_err_1 = delta_err[i];
				delta_err[i] = dx.mul((weights[i + 1]).t() * delta_err[i + 1]);
			}
		}
	}

	//Update weights
	void Net::updateWeights()
	{
		for (int i = 0; i < weights.size(); ++i)
		{
			cv::Mat delta_weights = learning_rate * (delta_err[i] * layer[i].t());
			cv::Mat delta_bias = learning_rate*delta_err[i];
			weights[i] = weights[i] + delta_weights;
			bias[i] = bias[i] + delta_bias;
		}
	}

	//Forward
	void Net::backward()
	{
		//move this function to the end of the forward().
		//calcLoss(layer[layer.size() - 1], target, output_error, loss);
		deltaError();
		updateWeights();
	}

	//Train,use accuracy_threshold
	void Net::train(cv::Mat input, cv::Mat target_, float accuracy_threshold)
	{
		if (input.empty())
		{
			std::cout << "Input is empty!" << std::endl;
			return;
		}

		std::cout << "Train,begain!" << std::endl;

		cv::Mat sample;
		if (input.rows == (layer[0].rows) && input.cols == 1)
		{
			target = target_;
			sample = input;
			layer[0] = sample;
			forward();
			//backward();
			int num_of_train = 0;
			while (accuracy < accuracy_threshold)
			{
				backward();
				forward();
				num_of_train++;
				if (num_of_train % 500 == 0)
				{
					std::cout << "Train " << num_of_train << " times" << std::endl;
					std::cout << "Loss: " << loss << std::endl;
				}
			}
			std::cout << std::endl << "Train " << num_of_train << " times" << std::endl;
			std::cout << "Loss: " << loss << std::endl;
			std::cout << "Train sucessfully!" << std::endl;
		}
		else if (input.rows == (layer[0].rows) && input.cols > 1)
		{
			double batch_loss = 0.;
			int epoch = 0;
			while (accuracy < accuracy_threshold)
			{
				batch_loss = 0.;
				for (int i = 0; i < input.cols; ++i)
				{
					target = target_.col(i);
					sample = input.col(i);

					layer[0] = sample;
					forward();
					batch_loss += loss;
					backward();
				}
				test(input, target_);
				epoch++;
				if (epoch % 10 == 0)
				{
					std::cout << "Number of epoch: " << epoch << std::endl;
					std::cout << "Loss sum: " << batch_loss << std::endl;
				}
				//if (epoch % 100 == 0)
				//{
				//	learning_rate*= 1.01;
				//}
			}
			std::cout << std::endl << "Number of epoch: " << epoch << std::endl;
			std::cout << "Loss sum: " << batch_loss << std::endl;
			std::cout << "Train sucessfully!" << std::endl;
		}
		else
		{
			std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
		}
	}

	//Train,use loss_threshold
	void Net::train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve)
	{
		if (input.empty())
		{
			std::cout << "Input is empty!" << std::endl;
			return;
		}

		std::cout << "Train,begain!" << std::endl;

		cv::Mat sample;
		if (input.rows == (layer[0].rows) && input.cols == 1)
		{
			target = target_;
			sample = input;
			layer[0] = sample;
			forward();
			//backward();
			int num_of_train = 0;
			while (loss > loss_threshold)
			{
				backward();
				forward();
				num_of_train++;
				if (num_of_train % 500 == 0)
				{
					std::cout << "Train " << num_of_train << " times" << std::endl;
					std::cout << "Loss: " << loss << std::endl;
				}
			}
			std::cout << std::endl << "Train " << num_of_train << " times" << std::endl;
			std::cout << "Loss: " << loss << std::endl;
			std::cout << "Train sucessfully!" << std::endl;
		}
		else if (input.rows == (layer[0].rows) && input.cols > 1)
		{
			double batch_loss = loss_threshold + 0.01;
			int epoch = 0;
			while (batch_loss > loss_threshold)
			{
				batch_loss = 0.;
				for (int i = 0; i < input.cols; ++i)
				{
					target = target_.col(i);
					sample = input.col(i);
					layer[0] = sample;

					forward();
					backward();

					batch_loss += loss;
				}

				loss_vec.push_back(batch_loss);

				if (loss_vec.size() >= 2 && draw_loss_curve)
				{
					draw_curve(board, loss_vec);
				}
				epoch++;
				if (epoch % output_interval == 0)
				{
					std::cout << "Number of epoch: " << epoch << std::endl;
					std::cout << "Loss sum: " << batch_loss << std::endl;
				}
				if (epoch % 100 == 0)
				{
					learning_rate *= fine_tune_factor;
				}
			}
			std::cout << std::endl << "Number of epoch: " << epoch << std::endl;
			std::cout << "Loss sum: " << batch_loss << std::endl;
			std::cout << "Train sucessfully!" << std::endl;
		}
		else
		{
			std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
		}
	}

	//Test
	void Net::test(cv::Mat &input, cv::Mat &target_)
	{
		if (input.empty())
		{
			std::cout << "Input is empty!" << std::endl;
			return;
		}
		std::cout << std::endl << "Predict,begain!" << std::endl;

		if (input.rows == (layer[0].rows) && input.cols == 1)
		{
			int predict_number = predict_one(input);

			cv::Point target_maxLoc;
			minMaxLoc(target_, NULL, NULL, NULL, &target_maxLoc, cv::noArray());		
			int target_number = target_maxLoc.y;

			std::cout << "Predict: " << predict_number << std::endl;
			std::cout << "Target:  " << target_number << std::endl;
			std::cout << "Loss: " << loss << std::endl;
		}
		else if (input.rows == (layer[0].rows) && input.cols > 1)
		{
			double loss_sum = 0;
			int right_num = 0;
			cv::Mat sample;
			for (int i = 0; i < input.cols; ++i)
			{
				sample = input.col(i);
				int predict_number = predict_one(sample);
				loss_sum += loss;

				target = target_.col(i);
				cv::Point target_maxLoc;
				minMaxLoc(target, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
				int target_number = target_maxLoc.y;

				std::cout << "Test sample: " << i << "   " << "Predict: " << predict_number << std::endl;
				std::cout << "Test sample: " << i << "   " << "Target:  " << target_number << std::endl << std::endl;
				if (predict_number == target_number)
				{
					right_num++;
				}
			}
			accuracy = (double)right_num / input.cols;
			std::cout << "Loss sum: " << loss_sum << std::endl;
			std::cout << "accuracy: " << accuracy << std::endl;
		}
		else
		{
			std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
			return;
		}
	}

	//Predict
	int Net::predict_one(cv::Mat &input)
	{
		if (input.empty())
		{
			std::cout << "Input is empty!" << std::endl;
			return -1;
		}

		if (input.rows == (layer[0].rows) && input.cols == 1)
		{
			layer[0] = input;
			forward();

			cv::Mat layer_out = layer[layer.size() - 1];
			cv::Point predict_maxLoc;

			minMaxLoc(layer_out, NULL, NULL, NULL, &predict_maxLoc, cv::noArray());
			return predict_maxLoc.y;
		}
		else
		{
			std::cout << "Please give one sample alone and ensure input.rows = layer[0].rows" << std::endl;
			return -1;
		}
	}

	//Predict,more  than one samples
	std::vector<int> Net::predict(cv::Mat &input)
	{
		cv::Mat sample;
		if (input.rows == (layer[0].rows) && input.cols > 1)
		{
			std::vector<int> predicted_labels;
			for (int i = 0; i < input.cols; ++i)
			{
				sample = input.col(i);
				int predicted_label = predict_one(sample);
				predicted_labels.push_back(predicted_label);
				return predicted_labels;
			}
		}
	}

	//Save model;
	void Net::save(std::string filename)
	{
		cv::FileStorage model(filename, cv::FileStorage::WRITE);
		model << "layer_neuron_num" << layer_neuron_num;
		model << "learning_rate" << learning_rate;
		model << "activation_function" << activation_function;

		for (int i = 0; i < weights.size(); i++)
		{
			std::string weight_name = "weight_" + std::to_string(i);
			model << weight_name << weights[i];
		}
		model.release();
	}

	//Load model;
	void Net::load(std::string filename)
	{
		cv::FileStorage fs;
		fs.open(filename, cv::FileStorage::READ);
		cv::Mat input_, target_;

		fs["layer_neuron_num"] >> layer_neuron_num;
		initNet(layer_neuron_num);

		for (int i = 0; i < weights.size(); i++)
		{
			std::string weight_name = "weight_" + std::to_string(i);
			fs[weight_name] >> weights[i];
		}

		fs["learning_rate"] >> learning_rate;
		fs["activation_function"] >> activation_function;

		fs.release();
	}

	//Get sample_number samples in XML file,from the start column. 
	void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start)
	{
		cv::FileStorage fs;
		fs.open(filename, cv::FileStorage::READ);
		cv::Mat input_, target_;
		fs["input"] >> input_;
		fs["target"] >> target_;
		fs.release();
		input = input_(cv::Rect(start, 0, sample_num, input_.rows));
		label = target_(cv::Rect(start, 0, sample_num, target_.rows));
	}

	//Draw loss curve
	void draw_curve(cv::Mat& board, std::vector<double> points)
	{
		cv::Mat board_(620, 1000, CV_8UC3, cv::Scalar::all(200));
		board = board_;
		cv::line(board, cv::Point(0, 550), cv::Point(1000, 550), cv::Scalar(0, 0, 0), 2);
		cv::line(board, cv::Point(50, 0), cv::Point(50, 1000), cv::Scalar(0, 0, 0), 2);

		for (size_t i = 0; i < points.size() - 1; i++)
		{
			cv::Point pt1(50 + i * 2, (int)(548 - points[i]));
			cv::Point pt2(50 + i * 2 + 1, (int)(548 - points[i + 1]));
			cv::line(board, pt1, pt2, cv::Scalar(0, 0, 255), 2);
			if (i >= 1000)
			{
				return;
			}
		}
		cv::imshow("Loss", board);
		cv::waitKey(1);
	}

}