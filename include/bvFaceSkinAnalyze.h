#pragma once

#ifdef FACESKINANALYZE
#define BV_EXPORTS __declspec(dllexport)
#else
#define BV_EXPORTS __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>

#define BV_DEBUG 1

class AnalyzeResult
{
public:
	int m_realAge;
	int m_faceType;
	cv::Mat m_img;
	cv::Mat m_parallel;
	cv::Mat m_cross;
	cv::Mat m_uv;

	float m_finalScore = 0;
	float m_skinAge = 0;

	cv::Rect m_faceDetectedRect;
	cv::Rect m_faceROIRect;
	std::vector < cv::Point2f> m_lmks;
	std::vector < cv::Point2f> m_fittedLmks;

	//cv::Mat m_skinMask;
	cv::Mat m_skinMaskFull;
	cv::Mat m_skinSpotMask;
	cv::Mat m_skinPoreMask;
	cv::Mat m_skinWrinkleMask;

	std::vector<cv::Point2i> m_wrinkle_spline_curve;
	std::vector<cv::Point2i> m_pore_spline_curve;
	std::vector<cv::Point2i> m_spot_spline_curve;


	/*****************cross**********************/
	cv::Mat m_predictRedMap;//预测红区图，这是个中间结果，为了后期定量计算红区所用，如果为空，定量计算红区也会生成
	cv::Mat m_redMap;// 红区图
	cv::Mat m_predictBrownMap;//预测棕区图，这是个中间结果，为了后期定量计算棕区所用，如果为空，定量计算棕区也会生成
	cv::Mat m_brownMap;//棕区图
	cv::Mat m_brownSpotAging;//色斑老化
	cv::Mat m_brownSpotRemoved;//色斑祛除
	cv::Mat m_dermisMat;// 真皮层

	int m_nCountSurfaceSpot;//表斑数量
	float m_areaSurfaceSpot;//表斑面积
	float m_scoreSurfaceSpot;//表斑得分

	int m_nCountRedSpot;//红区数量
	float m_areaRedSpot;//红区面积
	float m_scoreRedSpot;//红区得分

	cv::Mat m_resSurfaceSpot;
	cv::Mat m_resRedSpot;

	/****************daylight******************/
	cv::Mat m_wrinkleAging;//皱纹老化图
	cv::Mat m_visbleSpotRemoved;
	cv::Mat m_visbleSpotAging;
	int m_skinLevel = -1;//肤色

	/********************parallel********************/
	int m_nCountPore;//毛孔数量
	float m_areaPore;//毛孔面积
	float m_scorePore;//毛孔得分

	//cv::Mat m_poreMap;//毛孔图
	cv::Mat m_porePredictMap;//毛孔预测图，这是个中间结果，为了后期定量计算毛孔所用，如果为空，定量计算毛孔也会生成
	cv::Mat m_resPore;//毛孔结果图，png格式

	//WrinkleResult m_wr;//皱纹结果
	int m_nCountDeepWrinkle;
	int m_nCountLightWrinkle;

	int m_nCountLongWrinkle;
	int m_nCountShortWrinkle;

	float m_wrinkleResponse;
	float m_wrinkleArea;
	float m_wrinkleLength;
	float m_wrinkelScore;//皱纹得分
	cv::Mat m_resWrinkle;//皱纹结果图，png格式


	float m_resTexture;//粗造度
	float m_scoreTexture;//粗造度得分
	cv::Mat m_resMatTexture;//粗糙度影像

	
	
	

	/*************uv******************/
	cv::Mat m_resDeepSpot;
	cv::Mat m_uvSpotMap;//uv斑图
	int m_nCountDeepSpot;//uv版数量
	float m_areaDeepSpot;//uv斑面积
	float m_scoreDeepSpot;//uv斑得分

	cv::Mat m_resAcne;
	int m_nCountAcne;//油脂数量
	float m_areaAcne;//油脂面积
	float m_scoreAcne;//油脂得分
};


extern "C" BV_EXPORTS bool initFaceSkinAnalyze(const char* modelPath);

extern "C" BV_EXPORTS void genTemplate(cv::Mat& cross, cv::Mat& imageTemplate);

extern "C" BV_EXPORTS int faceSkinAnalyzeMultiTthead(cv::Mat& img, cv::Mat& parallel, cv::Mat& cross, cv::Mat& uv, bool bModifyImgs, AnalyzeResult& ar);
	
extern "C" BV_EXPORTS void releaseFaceSkinAnalyze();



/****************智能测量***************************/
extern "C" BV_EXPORTS cv::Mat Chamber3(cv::Mat& cross, float chamberRes[3]);

extern "C" BV_EXPORTS cv::Mat Eye5(cv::Mat& cross,  float eyeRes[5]);

extern "C" BV_EXPORTS cv::Mat FaceOutline(cv::Mat& cross,  float faceRes[5]);

extern "C" BV_EXPORTS cv::Mat MouthAngle(cv::Mat& cross,  float mouthRes[3], float& angle);

extern "C" BV_EXPORTS cv::Mat ChinNose(cv::Mat& cross, float& noseRes, float chinRes[3]);



/************模拟整形*******************************/
extern "C" BV_EXPORTS cv::Mat thinChin(cv::Mat& src, float strength);//0~100 尖下巴

extern "C" BV_EXPORTS cv::Mat thinNose(cv::Mat& src, float strength);//0~100 瘦鼻

extern "C" BV_EXPORTS cv::Mat thinQuangu(cv::Mat& src, float strength);//0~100 颧骨内推

extern "C" BV_EXPORTS cv::Mat thinFace(cv::Mat& src, float strength);//0~100 瘦脸

extern "C" BV_EXPORTS cv::Mat bigMouth(cv::Mat& src, float strength);//0~100 丰唇

/************眼眉处理*******************************/
extern "C" BV_EXPORTS cv::Mat lashAnalyze(cv::Mat& corss, float lashRes[4]);

extern "C" BV_EXPORTS cv::Mat browAnalyze(cv::Mat& corss, float browRes[10]);

extern "C" BV_EXPORTS cv::Mat eyesBagsAnalyze(cv::Mat& corss, cv::Mat& hor, float eyeBagRes[2]);

extern "C" BV_EXPORTS cv::Mat darkCircleAnalyze(cv::Mat& corss, float darkCircleRes[2]);
