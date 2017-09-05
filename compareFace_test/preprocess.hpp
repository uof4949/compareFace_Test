#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void load_cascade(CascadeClassifier& cascade, string fname) {
	String path = "C:/opencv/sources/data/haarcascades/"; // 검출기 폴더
	String full_name = path + fname;

	CV_Assert(cascade.load(full_name));
}

Mat preprocessing(Mat image) {
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY); // 명암도 영상변환
	equalizeHist(gray, gray);			// 히스토그램 평활화

	return gray;
}