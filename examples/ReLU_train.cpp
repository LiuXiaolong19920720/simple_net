#include"../include/Net.h"
//<opencv2\opencv.hpp>

using namespace std;
using namespace cv;
using namespace liu;

int main(int argc, char *argv[])
{
	//Set neuron number of every layer
	vector<int> layer_neuron_num = { 784,100,100,10 };

	// Initialise Net and weights
	Net net;
	net.initNet(layer_neuron_num);
	net.initWeights(0, 0., 0.01);

	//Get test samples and test samples 
	Mat input, label, test_input, test_label;
	int sample_number = 800;
	get_input_label("data/input_label_1-10_1000.xml", input, label, sample_number);
	get_input_label("data/input_label_1-10_1000.xml", test_input, test_label, 200, 800);

	//Set loss threshold,learning rate and activation function
	float loss_threshold = 412;
	net.learning_rate = 0.002;
	net.output_interval = 2;
	net.activation_function = "ReLU";

	//Train,and draw the loss curve(cause the last parameter is ture) and test the trained net
	net.train(input, label, loss_threshold, true);
	net.test(test_input, test_label);

	//Save the model
	net.save("models/model_ReLU_800_200.xml");

	getchar();
	return 0;

}




