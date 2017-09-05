#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Point2d calc_center(Rect obj) {				// �簢�� �߽� ���
	Point2d c = (Point2d)obj.size() / 2.0;
	Point2d center = (Point2d)obj.tl() + c;
	return center;
}

Mat calc_rotMap(Point2d face_center, vector<Point2d> pt) {
	Point2d delta = (pt[0].x > pt[1].x) ? pt[0] - pt[1] : pt[1] - pt[0];
	double angle = fastAtan2(delta.y, delta.x);					// �������� ���� ���

	Mat rot_mat = getRotationMatrix2D(face_center, angle, 1);
	return rot_mat;
}

Mat correct_image(Mat image, Mat rot_mat, vector<Point2d>& eyes_center) {
	Mat correct_img;
	warpAffine(image, correct_img, rot_mat, image.size(), INTER_CUBIC);

	for (int i = 0; i < eyes_center.size(); i++) {				// �� ��ǥ ȸ����ȯ
		Point3d coord(eyes_center[i].x, eyes_center[i].y, 1);	// ��İ����� 3���� ��ǥ��
		Mat dst = rot_mat * (Mat) coord;
		eyes_center[i] = (Point2f)dst;							// �� ��ǥ ����
	}
	return correct_img;											// ������� ��ȯ
}