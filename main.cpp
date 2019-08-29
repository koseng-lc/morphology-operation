/*
Author : Lintang E
NIM : 16/399897/TK/44911
*/

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

enum{
    EROSION = 1,
    DILATION = 0
};

inline uchar applyOp(const Mat &_input, int x, int y, const Mat &_kernel, int fits){
    int num_structuring_elem = countNonZero(_kernel);
    int count = 0;
    int map_x;
    int map_y;
    for(int i = -1; i < 2;i++){
        for(int j = -1; j < 2;j++){
            map_x = x+i;
            map_y = y+j;
            if(map_x < 0 || map_x >= _input.cols || map_y < 0 || map_y >= _input.rows)
                continue;
            if(_input.at<uchar>(map_y,map_x) > 0 && _kernel.at<uchar>(j+1,i+1) > 0)
                ++count;
        }
    }

    return fits > 0 ? static_cast<int>(count == num_structuring_elem) : static_cast<int>(count > 0);
}

Mat morphologicalOp(const Mat &_input, const Mat &_kernel, int morph_type){
    Mat input = _input.clone();
    Mat kernel = _kernel.clone();
    Mat output = Mat::zeros(input.size(), CV_8UC1);

    for(int i=0;i<input.rows;i++){
        for(int j=0;j<input.cols;j++){
            output.at<uchar>(i,j) = applyOp(input, j, i, kernel, morph_type) * 255;
        }
    }
    return output;
}

int main(){
    Mat input_img1 = imread("/media/koseng/4A7AE1C07AE1A941/lit/tpc/image 1.png",IMREAD_GRAYSCALE);
    Mat input_img2 = imread("/media/koseng/4A7AE1C07AE1A941/lit/tpc/image 2 (1).jpg",IMREAD_GRAYSCALE);

//    Mat numpang = imread("/media/koseng/New Volume1/wallpaper/456507-lake-water-forest-clear_sky-nature.jpg");
//    resize(numpang,numpang,Size(640,480));
//    imshow("RGB", numpang);
//    Mat cielab;
//    Mat ycrcb;
//    Mat yiq;
//    cvtColor(numpang,cielab,CV_BGR2Lab);
//    cvtColor(numpang,ycrcb,CV_BGR2YCrCb);
//    cvtColor(numpang,yiq,CV_BGR2HSV);
//    imshow("RGB",numpang);
//    imshow("ycrcb",ycrcb);
//    imshow("yiq",yiq);

    Mat kernel = (Mat_<uchar>(3,3) << 0, 1, 0, 1, 1, 1, 0, 1, 0); // ELLIPSE STRUCTURING ELEMENT

    cout << kernel << endl;

    //Apply top-hat transformation
    Mat output1 = morphologicalOp(input_img1, kernel, EROSION);
    output1 = morphologicalOp(output1, kernel, DILATION);
    output1 = input_img1 - output1;

    Mat output2 = morphologicalOp(input_img2, kernel, EROSION);
    output2 = morphologicalOp(output2, kernel, DILATION);
    output2 = input_img2 - output2;

    //Apply bottom-hat transformation
    Mat output3 = morphologicalOp(input_img1, kernel, DILATION);
    output3 = morphologicalOp(output3, kernel, EROSION);
    output3 = output3 - input_img1;

    Mat output4 = morphologicalOp(input_img2, kernel, DILATION);
    output4 = morphologicalOp(output4, kernel, EROSION);
    output4 = output4 - input_img2;

    vector<vector<Point> > contours;
    Mat output5 = Mat::zeros(input_img1.size(), CV_8UC1);
    findContours(input_img1, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//get the outest and authentic contours
    for(size_t i =0;i<contours.size();i++){
        vector<Point > cvx_pt;
        vector<vector<Point> > draw_pts;
        convexHull(contours[i], cvx_pt);
        draw_pts.push_back(cvx_pt);
        drawContours(output5,draw_pts,0,Scalar(255),1);
    }

    contours.clear();

    Mat output6 = Mat::zeros(input_img2.size(), CV_8UC1);
    findContours(input_img2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//get the outest and authentic contours
    for(size_t i =0;i<contours.size();i++){
        vector<Point > cvx_pt;
        vector<vector<Point> > draw_pts;
        convexHull(contours[i], cvx_pt);
        draw_pts.push_back(cvx_pt);
        drawContours(output6,draw_pts,0,Scalar(255),1);
    }

    imshow("Input1", input_img1);
    imshow("Input2", input_img2);
    imshow("Output1 - TOP-HAT", output1);
    imshow("Output2 - TOP-HAT", output2);
    imshow("Output1 - BOTTOM-HAT", output3);
    imshow("Output2 - BOTTOM-HAT", output4);
    imshow("Output1 - Convex Hull", output5);
    imshow("Output2 - Convex Hull", output6);
    waitKey(0);
    return 0;
}
