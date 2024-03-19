#include <opencv2/opencv.hpp>
#include <iostream>  
#include "bvFaceSkinAnalyze.h"

using namespace cv;
using namespace std;


int main()
{	
	const char* modelPth = "models";	
	bool bInit = initFaceSkinAnalyze(modelPth);
	if (!bInit)
	{
		std::cout << "init model failed!" << std::endl;
		return -1;
	}
	
	cv::Mat img = cv::imread("imgs//daylight (571).jpg", 1);
	cv::Mat img_cross = cv::imread("imgs//cross (571).jpg", 1);
	cv::Mat img_uv = cv::imread("imgs//uv (571).jpg", 1);
	cv::Mat img_hor = cv::imread("imgs//parallel (571).jpg", 1);
    
    for (int i = 0; i < 1; i++)
    {
        cv::Mat templateMat;
        genTemplate(img_cross, templateMat);
        imwrite("templateMat.png", templateMat);


        AnalyzeResult _ar;
        _ar.m_realAge = 34;
        faceSkinAnalyzeMultiTthead(img, img_hor, img_cross, img_uv, 1, _ar);


        if (_ar.m_faceType == 1)
        {
            /**********眼眉***************/
            float darkCircleRes[2];
            Mat resDarkCircle = darkCircleAnalyze(img_cross, darkCircleRes);
            cout << "darkCircleAnalyze " << darkCircleRes[0] << " " << darkCircleRes[1] << endl;
            imwrite("resDarkCircle.png", resDarkCircle);
            resDarkCircle.release();

            float eyeBageRes[2];
            Mat resEyeBags = eyesBagsAnalyze(img_cross, img_hor, eyeBageRes);
            cout << "eyesBagsAnalyze " << eyeBageRes[0] << " " << eyeBageRes[1] << endl;
            imwrite("resEyeBags.png", resEyeBags);
            resEyeBags.release();

            float lashRes[4];
            Mat resLash = lashAnalyze(img_cross, lashRes);
            cout << "lashRes " << lashRes[0] << " " << lashRes[1] << " " << lashRes[2] << " " << lashRes[3] << endl;
            imwrite("resLash.png", resLash);
            resLash.release();

            float brownRes[10];
            Mat resBrow = browAnalyze(img_cross, brownRes);
            for (int i = 0; i < 10; i++)
            {
                cout << brownRes[i] << endl;
            }
            imwrite("resBrow.png", resBrow);
            resBrow.release();

            /**********智能测量****************/
            float valueChamber3[3];
            Mat resPngChamber = Chamber3(img_cross, valueChamber3);
            imwrite("resPngChamber.png", resPngChamber);


            float eye5MatRes[5];
            Mat eye5Mat = Eye5(img_cross, eye5MatRes);
            imwrite("eye5Mat.png", eye5Mat);



            float faceOutlineRes[5];
            Mat faceOutlineMat = FaceOutline(img_cross, faceOutlineRes);
            imwrite("faceOutlineMat.png", faceOutlineMat);

            float gAngle = 0;
            float mouthRes[3];
            Mat MouthAngleMat = MouthAngle(img_cross, mouthRes, gAngle);
            imwrite("MouthAngleMat.png", MouthAngleMat);

            float noseRes = 0;
            float chinRes[3];
            Mat ChinNoseMat = ChinNose(img_cross, noseRes, chinRes);
            imwrite("chinNoseMat.png", ChinNoseMat);


            /***********模拟整形*****************/
            Mat newNose = thinNose(img_cross, 40.);
            Mat newChin = thinChin(img_cross, 40.);
            Mat newQuangu = thinQuangu(img_cross, 40.);
            Mat newFace = thinFace(img_cross, 40.);
            Mat newMouth = bigMouth(img_cross, 40.);

            imwrite("newNose.jpg", newNose);
            imwrite("newChin.jpg", newChin);
            imwrite("newQuangu.jpg", newQuangu);
            imwrite("newFace.jpg", newFace);
            imwrite("newMouth.jpg", newMouth);
        }
    }
    
    
    releaseFaceSkinAnalyze();
	return 0;
}