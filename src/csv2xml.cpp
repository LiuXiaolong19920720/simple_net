#include<opencv2\opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;


int csv2xml()
//int main()
{
	CvMLData mlData;
	mlData.read_csv("train.csv");//¶ÁÈ¡csvÎÄ¼þ
	Mat data = cv::Mat(mlData.get_values(), true);
	cout << "Data have been read successfully!" << endl;
	//Mat double_data;
	//data.convertTo(double_data, CV_64F);
	
	Mat input_ = data(Rect(1, 1, 784, data.rows - 1)).t();
	Mat label_ = data(Rect(0, 1, 1, data.rows - 1));
	Mat target_(10, input_.cols, CV_32F, Scalar::all(0.));

	Mat digit(28, 28, CV_32FC1);
	Mat col_0 = input_.col(3);
	float label0 = label_.at<float>(3, 0);
	cout << label0;
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			digit.at<float>(i, j) = col_0.at<float>(i * 28 + j);
		}
	}

	for (int i = 0; i < label_.rows; ++i)
	{
		float label_num = label_.at<float>(i, 0);
		//target_.at<float>(label_num, i) = 1.;
		target_.at<float>(label_num, i) = label_num;
	}

	Mat input_normalized(input_.size(), input_.type());
	for (int i = 0; i < input_.rows; ++i)
	{
		for (int j = 0; j < input_.cols; ++j)
		{
			//if (input_.at<double>(i, j) >= 1.)
			//{
			input_normalized.at<float>(i, j) = input_.at<float>(i, j) / 255.;
			//}
		}
	}

	string filename = "input_label_0-9.xml";
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "input" << input_normalized;
	fs << "target" << target_; // Write cv::Mat
	fs.release();


	Mat input_1000 = input_normalized(Rect(0, 0, 10000, input_normalized.rows));
	Mat target_1000 = target_(Rect(0, 0, 10000, target_.rows));

	string filename2 = "input_label_0-9_10000.xml";
	FileStorage fs2(filename2, FileStorage::WRITE);

	fs2 << "input" << input_1000;
	fs2 << "target" << target_1000; // Write cv::Mat
	fs2.release();

	return 0;
}
