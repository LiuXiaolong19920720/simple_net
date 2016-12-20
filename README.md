##simple_net
**Simple net** is a simple deep neural network implemented in C++，based with OpenCV Mat matrix class

---

## Examples
You can initialize a neural network just like bellow:
```cpp
	//Set neuron number of every layer
	vector<int> layer_neuron_num = { 784,100,10 };

	// Initialise Net and weights
	Net net;
	net.initNet(layer_neuron_num);
	net.initWeights(0, 0., 0.01);
	net.initBias(Scalar(0.5));
```

It is very easy to train:

```cpp
#include"../include/Net.h"
//<opencv2\opencv.hpp>

using namespace std;
using namespace cv;
using namespace liu;

int main(int argc, char *argv[])
{
	//Set neuron number of every layer
	vector<int> layer_neuron_num = { 784,100,10 };

	// Initialise Net and weights
	Net net;
	net.initNet(layer_neuron_num);
	net.initWeights(0, 0., 0.01);
	net.initBias(Scalar(0.5));

	//Get test samples and test samples 
	Mat input, label, test_input, test_label;
	int sample_number = 800;
	get_input_label("data/input_label_1000.xml", input, label, sample_number);
	get_input_label("data/input_label_1000.xml", test_input, test_label, 200, 800);

	//Set loss threshold,learning rate and activation function
	float loss_threshold = 0.5;
	net.learning_rate = 0.3;
	net.output_interval = 2;
	net.activation_function = "sigmoid";

	//Train,and draw the loss curve(cause the last parameter is ture) and test the trained net
	net.train(input, label, loss_threshold, true);
	net.test(test_input, test_label);

	//Save the model
	net.save("models/model_sigmoid_800_200.xml");

	getchar();
	return 0;
}
```

It is easier to load a trained net and use:
```cpp
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

	//Load the trained net and test.
	Net net;
	net.load("models/model_sigmoid_800_200.xml");
	net.test(test_input, test_label);

	getchar();
	return 0;
}
```




