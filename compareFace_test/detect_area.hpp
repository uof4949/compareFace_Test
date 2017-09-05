#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Rect detect_lip(Point2d face_center, Rect face) {
	Point2d lip_center = face_center + Point2d(0, face.height * 0.30);		// 입술 중심
	Point2d gap_size(face.width * 0.18, face.height * 0.1);					// 정해진 크기 - 평행이동 거리

	Point lip_start = lip_center - gap_size;								// 왼쪽 위로 평행이동
	Point lip_end = lip_center + gap_size;									// 오른쪽 아래로 평행이동

	return Rect(lip_start, lip_end);										// 입술 사각형
}

void detect_hair(Point2d face_center, Rect face, vector<Rect> &hair_rect) {
	Point2d h_gap(face.width * 0.45, face.height * 0.65);					// 평행이동 거리
	Point2d pt1 = face_center - h_gap;										// 왼쪽 상단 평행이동
	Point2d pt2 = face_center + h_gap;										// 오른쪽 상단 평행이동
	Rect hair(pt1, pt2);													// 전체 머리영역

	Size size(hair.width, hair.height * 0.40);								// 윗머리 및 귀밑머리 크기
	Rect hair1(hair.tl(), size);											// 윗머리 영역
	Rect hair2(hair.br() - (Point)size, size);								// 귀밑머리 영역

	hair_rect.push_back(hair1);												// Rect형 벡터에 저장
	hair_rect.push_back(hair2);									
	hair_rect.push_back(hair);												// 전체머리영역을 마지막에 저장
}