#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include<QFileDialog>
#include <QSqlQuery>
#include <QSqlDatabase>
#include <featuredata.h>
#include <display_image.h>

using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    Mat object;
    explicit MainWindow(QWidget *parent = 0);   
    QSqlDatabase Database;
    Display_Image disp;
    QSqlQuery *query;
    int frameNum;int CurrentFrame;
    VideoCapture cap;
    Mat3b frame1,frame3, result_image;
    QImage image1, image2;
    FeatureData featureData;

    void FAST_corner_detection();
    void SIFT_feature_detection();
    void SURF_feature_detection();
    void BRIEF_feature_detection();
    void ORB_feature_detection();
    void addDetector_to_pop_up_manu();

    void DisplayDetectedObj();
    void Classifier_selection();
    void Feature_selection();
    void Classifier_Apply();
    void DisplayImages();

    QImage Mat3b2QImage (Mat3b src);


    double nFeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma;

    vector<Rect> faces;
    vector<Mat>collect;
    Mat inputImg_1, inputImg_2, frame, frame2, object_detected_image;
    QImage image;
    QString getData_1,getData_2;
    QPoint point2;
    CascadeClassifier cascade;
    std::string classifier, image_path_detect,image_path_track, video_path_track;
    ~MainWindow();

private slots:
    void on_actionExit_triggered();

    void on_actionDetect_triggered();

    void on_actionVideo_triggered();

    void on_actionTrack_triggered();

    void on_RUN_clicked();       

    void on_detector_clicked();

    void on_Detectors_currentIndexChanged(const QString &arg1);    

    void on_spinBox_valueChanged(int arg1);

    void on_actionContents_triggered();

    void on_actionAbout_triggered();

    void on_pushButton_clicked();

    void on_Quit_clicked();

private:
    Ui::MainWindow *ui;

public slots:
    void showMousePosition(QPoint& pos);
signals:
    void sendCurrentFrame(int);
};

#endif // MAINWINDOW_H
