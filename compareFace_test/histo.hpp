#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void calc_Histo(const Mat& img, Mat& hist, Vec3i bins, Vec3f range, Mat mask) {
	int dims = img.channels();							// 히스토그램 차원 수
	int channels[] = { 0, 1, 2 };						// 채널 목록
	int histSize[] = { bins[0], bins[1], bins[2] };

	float range1[] = { 0, range[0] };
	float range2[] = { 0, range[1] };
	float range3[] = { 0, range[2] };
	const float* ranges[] = { range1, range2, range3 }; // 모든 채널 화소범위

	calcHist(&img, 1, channels, mask, hist, dims, histSize, ranges);
}

void draw_ellipse(Mat& image, Rect2d obj, Scalar color, int thickness, float ratio) {
	Point2d center = obj.tl() + (Point2d)obj.size() * 0.5;		// 타원 중심
	Size2d size = (Size2d)obj.size() * 0.45;					// 타원 크기
	ellipse(image, center, size, 0, 0, 360, color, thickness);
}

void make_masks(vector<Rect> sub_obj, Size org_size, Mat mask[4]) {
	Mat base_mask(org_size, CV_8U, Scalar(0));						// 기본마스크
	draw_ellipse(base_mask, sub_obj[2], Scalar(255), -1, 0.45f);	// 기본 마스크에 타원그리기
	
	mask[0] = base_mask(sub_obj[0]);								// 윗머리 마스크
	mask[1] = base_mask(sub_obj[1]);								// 귀밑머리 마스크

	draw_ellipse(base_mask, sub_obj[3], Scalar(0), -1, 0.45f);
	mask[3] = base_mask(sub_obj[3]);								// 입술 마스크
	mask[2] = base_mask(sub_obj[2]);								// 얼굴 마스크
}

void calc_histos(Mat correct_img, vector<Rect> sub_obj, Mat hists[4], Mat masks[4]) {
	Vec3i bins(64, 64, 64);									// 히스토그램 계급 개수
	Vec3f ranges(256, 256, 256);							// 각 채널 화소범위

	for (int i = 0; i < (int)sub_obj.size(); i++) {
		Mat sub = correct_img(sub_obj[i]);					// 검출 서브 영상
		calc_Histo(sub, hists[i], bins, ranges, masks[i]);	// 히스토그램 계산
		// imshow(format("mask[%d]", i), masks[i]);			// 마스크 영상 출력
		// waitKey();
	}
}