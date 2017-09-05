#include "preprocess.hpp"								// �����ε� �� ��ó�� �Լ�
#include "correct_angle.hpp"							// ���� ���� �Լ���
#include "detect_area.hpp"								// �Լ� �� �Ӹ����� ���� �Լ�
#include "histo.hpp"									// ������׷� �� ���� �Լ�

int main() {
	// ���� 11.3.2�� 5 ~ 16��
	CascadeClassifier face_cascade, eyes_cascade;
	load_cascade(face_cascade, "haarcascade_frontalface_alt2.xml");
	load_cascade(eyes_cascade, "haarcascade_eye.xml");

	//Mat image1 = imread("../face/59.jpg", IMREAD_COLOR);
	//Mat image2 = imread("../face/60.jpg", IMREAD_COLOR);
	//Mat image1 = imread("../face/a.png", IMREAD_COLOR);
	//Mat image2 = imread("../face/b.png", IMREAD_COLOR);
	//Mat image1 = imread("../face/c.png", IMREAD_COLOR);
	//Mat image2 = imread("../face/d.png", IMREAD_COLOR);
	//Mat image1 = imread("../face/e.jpg", IMREAD_COLOR);
	//Mat image2 = imread("../face/f.jpg", IMREAD_COLOR);
	Mat image1 = imread("../face/g.jpg", IMREAD_COLOR);
	Mat image2 = imread("../face/h.jpg", IMREAD_COLOR);
	CV_Assert(image1.data);
	CV_Assert(image2.data);
	Mat gray1 = preprocessing(image1);			// ��ó��
	Mat gray2 = preprocessing(image2);			// ��ó��

	vector<Rect> faces1, faces2, eyes1, eyes2, sub_obj1, sub_obj2;
	vector<Point2d> eyes_center1, eyes_center2;

	face_cascade.detectMultiScale(gray1, faces1, 1.1, 2, 0, Size(100, 100));	// �� ����
	face_cascade.detectMultiScale(gray2, faces2, 1.1, 2, 0, Size(100, 100));	// �� ����

	if ((faces1.size() > 0) && (faces2.size() > 0)) {
		eyes_cascade.detectMultiScale(gray1(faces1[0]), eyes1, 1.15, 7, 0, Size(25, 20));
		eyes_cascade.detectMultiScale(gray2(faces2[0]), eyes2, 1.15, 7, 0, Size(25, 20));

		if ((eyes1.size() == 2) && (eyes2.size() == 2)) {
			eyes_center1.push_back(calc_center(eyes1[0] + faces1[0].tl()));
			eyes_center2.push_back(calc_center(eyes2[1] + faces2[0].tl()));

			Point2d face_center1 = calc_center(faces1[0]);
			Point2d face_center2 = calc_center(faces2[0]);

			Mat rot_mat1 = calc_rotMap(face_center1, eyes_center1);
			Mat rot_mat2 = calc_rotMap(face_center2, eyes_center2);

			Mat correct_img1 = correct_image(image1, rot_mat1, eyes_center1);		// ���� ����
			Mat correct_img2 = correct_image(image2, rot_mat2, eyes_center2);		// ���� ����

			detect_hair(face_center1, faces1[0], sub_obj1);						// �Ӹ������� ����
			detect_hair(face_center2, faces2[0], sub_obj2);						// �Ӹ������� ����

			sub_obj1.push_back(detect_lip(face_center1, faces1[0]));				// �Լ� ����
			sub_obj2.push_back(detect_lip(face_center2, faces2[0]));				// �Լ� ����

			Mat masks1[4], masks2[4], hists1[4], hists2[4];

			make_masks(sub_obj1, correct_img1.size(), masks1);						// 4�� ����ũ ����
			make_masks(sub_obj2, correct_img2.size(), masks2);						// 4�� ����ũ ����

			calc_histos(correct_img1, sub_obj1, hists1, masks1);					// 4�� ������׷� ����
			calc_histos(correct_img2, sub_obj2, hists2, masks2);					// 4�� ������׷� ����

																				// ������׷� �� - ���絵 �� ���
			double criteria1 = compareHist(hists1[0], hists2[0], CV_COMP_CORREL);		// ���Ӹ� ��
			double criteria2 = compareHist(hists1[1], hists2[1], CV_COMP_CORREL);		// �͹ظӸ� ��
			double criteria3 = compareHist(hists1[2], hists2[2], CV_COMP_CORREL);		// �Լ� ��
			double criteria4 = compareHist(hists1[3], hists2[3], CV_COMP_CORREL);		// ��ü ��

			rectangle(image1, sub_obj1[0], Scalar(255, 0, 0), 1);							// ���Ӹ� ���� �簢�� �׸���
			rectangle(image2, sub_obj2[0], Scalar(255, 0, 0), 1);							// ���Ӹ� ���� �簢�� �׸���

			rectangle(image1, sub_obj1[1], Scalar(255, 0, 0), 2);							// �͹ظӸ� ���� �簢�� �׸���
			rectangle(image2, sub_obj2[1], Scalar(255, 0, 0), 2);							// �͹ظӸ� ���� �簢�� �׸���

			rectangle(image1, sub_obj1[2], Scalar(255, 0, 0), 3);							// ��ü ���� �簢�� �׸���
			rectangle(image2, sub_obj2[2], Scalar(255, 0, 0), 3);							// ��ü ���� �簢�� �׸���

			rectangle(image1, sub_obj1[3], Scalar(255, 0, 0), 4);							// �Լ� ���� �簢�� �׸���
			rectangle(image2, sub_obj2[3], Scalar(255, 0, 0), 4);							// �Լ� ���� �簢�� �׸���

			cout << format("���Ӹ� ���絵 %4.2f\n", criteria1);
			cout << format("�͹ظӸ� ���絵 %4.2f\n", criteria2);
			cout << format("��ü ���絵 %4.2f\n", criteria3);
			cout << format("�Լ� ���絵 %4.2f\n", criteria4);

			imshow("image1", image1);
			imshow("image2", image2);

			waitKey(0);
		}
	}
	else {
		cout << "Error!\n";
	}
	return 0;
}