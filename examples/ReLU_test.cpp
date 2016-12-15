#include"../include/Net.h"
//<opencv2\opencv.hpp>

using namespace std;
using namespace cv;
using namespace liu;

int main(int argc, char *argv[])
{
	//Get test samples and the label is 0--9
	Mat test_input, test_label;
	int sample_number = 200;
	int start_position = 800;
	get_input_label("data/input_label_0-9_1000.xml", test_input, test_label, sample_number, start_position);

	//Load the trained net and test.
	Net net;
	net.load("models/model_ReLU_800_200.xml");
	net.test(test_input, test_label);

	getchar();
	return 0;
}




