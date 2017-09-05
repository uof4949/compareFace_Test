#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Point2d calc_center(Rect obj) {				// 사각형 중심 계산
	Point2d c = (Point2d)obj.size() / 2.0;
	Point2d center = (Point2d)obj.tl() + c;
	return center;
}

Mat calc_rotMap(Point2d face_center, vector<Point2d> pt) {
	Point2d delta = (pt[0].x > pt[1].x) ? pt[0] - pt[1] : pt[1] - pt[0];
	double angle = fastAtan2(delta.y, delta.x);					// 차분으로 기울기 계산

	Mat rot_mat = getRotationMatrix2D(face_center, angle, 1);
	return rot_mat;
}

Mat correct_image(Mat image, Mat rot_mat, vector<Point2d>& eyes_center) {
	Mat correct_img;
	warpAffine(image, correct_img, rot_mat, image.size(), INTER_CUBIC);

	for (int i = 0; i < eyes_center.size(); i++) {				// 눈 좌표 회전변환
		Point3d coord(eyes_center[i].x, eyes_center[i].y, 1);	// 행렬곱위해 3차원 좌표로
		Mat dst = rot_mat * (Mat) coord;
		eyes_center[i] = (Point2f)dst;							// 눈 좌표 저장
	}
	return correct_img;											// 보정결과 반환
}