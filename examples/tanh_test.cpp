#include"../include/Net.h"
//<opencv2\opencv.hpp>

using namespace std;
using namespace cv;
using namespace liu;

int main(int argc, char *argv[])
{
	//Get test samples and the label is 0--1
	Mat test_input, test_label;
	int sample_number = 200;
	int start_position = 800;
	get_input_label("data/input_label_1000.xml", test_input, test_label, sample_number, start_position);

	//convert label from 0---1 to -1---1,cause tanh function range is [-1,1]
	test_label = 2 * test_label - 1;

	//Load the trained net and test.
	Net net;
	net.load("models/model_tanh_800_200.xml");
	net.test(test_input, test_label);

	getchar();
	return 0;
}




