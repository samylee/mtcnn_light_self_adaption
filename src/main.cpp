#include "network.h"
#include "mtcnn.h"
#include <time.h>
#include <opencv2/contrib/contrib.hpp>

int main()
{
	//initial models without image's width or height
	mtcnn find;

	//test_type choice
	string test_type = "img_dir";
	if (test_type == "img_dir")
	{
		Directory dir;
		string imgpath = "images_adaption/";
		vector<string> filenames = dir.GetListFiles(imgpath, "*.jpg", false);
		for (int i = 0; i < filenames.size(); i++)
		{
			string imgname = imgpath + filenames[i];
			Mat image = imread(imgname, 1);

			clock_t start_t = clock();
			//detect face by min_size(30)
			find.findFace(image, 30);
			cout << "Cost time: " << clock() - start_t << endl;

			imshow("test", image);
			waitKey(0);
		}
	}
	else if(test_type == "video")
	{
		VideoCapture cap("test.mp4");
		if (!cap.isOpened())
			cout << "fail to open!" << endl;
		Mat image;

		while (true) {
			
			cap >> image;
			if (!image.data) {
				cout << "fail to read image!" << endl;
				return -1;
			}

			clock_t start_t = clock();
			//detect face by min_size(60)
			find.findFace(image, 60);
			cout << "Cost time: " << clock() - start_t << endl;

			imshow("result", image);
			if (waitKey(1) >= 0) break;
		}
	}
	else
	{
		cout << "Unknow test type!" << endl;
	}

	return 0;
}