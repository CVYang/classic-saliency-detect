#include "tracking.h"
using namespace cv;
using namespace std;
Tracking::Tracking(){}
Tracking::~Tracking(){}
// void Tracking::init(){
//     hCamera = mVision.init();
//     mVision.startPlay(hCamera);
// }
RotatedRect adjustRRect(const RotatedRect & rect){
    const Size2f & s = rect.size;
    if (s.width < s.height)
        return rect;
    return RotatedRect(rect.center, Size2f(s.height, s.width), rect.angle + 90.0);
}

//返回两点欧式距离
float distance(Point2f point1,Point2d point2){
    return sqrt(pow(point2.x-point1.x,2)+pow(point2.y-point1.y,2));
}


Mat SalientRegionDetectionBasedonLC(Mat &src){
	int HistGram[256]={0};
	int row=src.rows,col=src.cols;
	int gray[row][col];    //height*width
	//int Sal_org[row][col];
	int val;
	Mat Sal=Mat::zeros(src.size(),CV_8UC1 );
	Point3_<uchar>* p;   //三维通道指针
	for (int i=0;i<row;i++){
		for (int j=0;j<col;j++){
			p=src.ptr<Point3_<uchar> > (i,j);
			val=(p->x + (p->y) *2 + p->z)/4;
			HistGram[val]++;
			gray[i][j]=val;
		}
	}
	int Dist[256];
	int Y,X;
	int max_gray=0;
	int min_gray=1<<28;
	for (Y = 0; Y < 256; Y++)
    {
        val = 0;
        for (X = 0; X < 256; X++) 
            val += abs(Y - X) * HistGram[X];                //    论文公式（9），灰度的距离只有绝对值，这里其实可以优化速度，但计算量不大，没必要了
        Dist[Y] = val;
        max_gray=max(max_gray,val);
        min_gray=min(min_gray,val);
    }
 
    
    for (Y = 0; Y < row; Y++)
    {
        for (X = 0; X < col; X++)
        {
            Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]] - min_gray)*255/(max_gray - min_gray);        //    计算全图每个像素的显著性
        	//Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]])*255/(max_gray);        //    计算全图每个像素的显著性
        
        }
    }
    return Sal;
 
}

Mat SalientRegionDetectionBasedonAC(Mat &src,int MinR2, int MaxR2,int Scale){
	Mat Lab;
	cvtColor(src, Lab, CV_BGR2Lab); 
 
	int row=src.rows,col=src.cols;
	int Sal_org[row][col];
	memset(Sal_org,0,sizeof(Sal_org));  
    //void *memset(void *s,int c,size_t n)
    //总的作用：将已开辟内存空间 s 的首 n 个字节的值设为值 c
	
	Mat Sal=Mat::zeros(src.size(),CV_8UC1 );
 
	Point3_<uchar>* p;
	Point3_<uchar>* p1;
	int val;
	Mat filter;
 
	int max_v=0;
	int min_v=1<<28;  //为啥左移28位？
	for (int k=0;k<Scale;k++){
		int len=(MaxR2 - MinR2) * k / (Scale - 1) + MinR2;
		blur(Lab, filter, Size(len,len ));
		for (int i=0;i<row;i++){
			for (int j=0;j<col;j++){
				p=Lab.ptr<Point3_<uchar> > (i,j);
				p1=filter.ptr<Point3_<uchar> > (i,j);
				//cout<<(p->x - p1->x)*(p->x - p1->x)+ (p->y - p1->y)*(p->y-p1->y) + (p->z - p1->z)*(p->z - p1->z) <<" ";
				
				val=sqrt( (p->x - p1->x)*(p->x - p1->x)+ (p->y - p1->y)*(p->y-p1->y) + (p->z - p1->z)*(p->z - p1->z) );
				Sal_org[i][j]+=val;
				if(k==Scale-1){
					max_v=max(max_v,Sal_org[i][j]);
					min_v=min(min_v,Sal_org[i][j]);
				}
			}
		}
}
	
	//cout<<max_v<<" "<<min_v<<endl;
	int X,Y;
    for (Y = 0; Y < row; Y++)
    {
        for (X = 0; X < col; X++)
        {
            Sal.at<uchar>(Y,X) = (Sal_org[Y][X] - min_v)*255/(max_v - min_v);        //    计算全图每个像素的显著性
        	//Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]])*255/(max_gray);        //    计算全图每个像素的显著性
        }
    }
    return Sal;
}

Mat SalientRegionDetectionBasedonFT(Mat &src){
	Mat Lab;
	cvtColor(src, Lab, CV_BGR2Lab); 
 
	int row=src.rows,col=src.cols;
 
	int Sal_org[row][col];
	memset(Sal_org,0,sizeof(Sal_org));
	
	Point3_<uchar>* p;
 
	int MeanL=0,Meana=0,Meanb=0;
	for (int i=0;i<row;i++){
		for (int j=0;j<col;j++){
			p=Lab.ptr<Point3_<uchar> > (i,j);
			MeanL+=p->x;
			Meana+=p->y;
			Meanb+=p->z;
		}
	}
	MeanL/=(row*col);
	Meana/=(row*col);
	Meanb/=(row*col);
 
	GaussianBlur(Lab,Lab,Size(3,3),0,0);
 
	Mat Sal=Mat::zeros(src.size(),CV_8UC1 );
 
	int val;
 
	int max_v=0;
	int min_v=1<<28;
 
	for (int i=0;i<row;i++){
		for (int j=0;j<col;j++){
			p=Lab.ptr<Point3_<uchar> > (i,j);
			val=sqrt( (MeanL - p->x)*(MeanL - p->x)+ (p->y - Meana)*(p->y-Meana) + (p->z - Meanb)*(p->z - Meanb) );
			Sal_org[i][j]=val;
			max_v=max(max_v,val);
			min_v=min(min_v,val);		
		}
	}
	

	int X,Y;
    for (Y = 0; Y < row; Y++)
    {
        for (X = 0; X < col; X++)
        {
            Sal.at<uchar>(Y,X) = (Sal_org[Y][X] - min_v)*255/(max_v - min_v);        //    计算全图每个像素的显著性
        	//Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]])*255/(max_gray);        //    计算全图每个像素的显著性
        }
    }
    return Sal;
}


int main(){
    //VideoCapture cap1("../log_color_8.avi");
    //VideoCapture cap1("/home/vayneli/test_video/0327/8_color.avi");
    VideoCapture cap1("/home/vayneli/tracking_big/log_color_8.avi");
    static VideoWriter writer;
    writer.open("rectangle.avi",CV_FOURCC('M','J','P','G'),25,Size(640,480),1);
    // VideoCapture cap2(3);
    // cout<<"o"<<endl;
    if(!cap1.isOpened())
    {
        return -1;
    }
    Rect roi;
    roi.x = 0;
    roi.y = 100;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarcy;
    cv::Mat img_src1;
    Mat result;
    //获得帧率
    //int count = 1;
    vector<Rect> unknown;
    Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
    //int x=-1;
    while(1){
        Rect temp;
        cap1>>img_src1;
        if(img_src1.empty()) return 0;
        roi.width = img_src1.cols;
        //cout<<img_src1.cols<<endl;
        //cout<<img_src1.rows<<endl;
        roi.height = 300;
        Mat src = img_src1(roi);
        //result = SalientRegionDetectionBasedonLC(src);
        //result = SalientRegionDetectionBasedonFT(src);
        result=SalientRegionDetectionBasedonAC(src,src.rows/8,src.rows/2,3);
        threshold(result, result, 40, 255, THRESH_BINARY);
        morphologyEx(result,result,MORPH_CLOSE,element);
        findContours(result, contours, hierarcy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for(size_t i=0;i<contours.size();i++){
            if(contours[i].size()>5){
                temp = boundingRect(contours[i]);
                temp.y+=roi.y;
                if(temp.area()>2000) continue;
                if(temp.area()<200) continue;
                rectangle(img_src1,temp,Scalar(255,255,0));
            }
        }
        imshow("src",result);
        imshow("src1",img_src1);
        //img_src1.convertTo(img_src1,CV_8UC3);
        writer.write(img_src1);
        if( cvWaitKey(1) >= 0 ) break; 
    }
    writer.release();
    
}
