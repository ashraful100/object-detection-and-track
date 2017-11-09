#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <QtCore>
#include <QtGui>
#include <QMessageBox>
#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <QMessageBox>
#include <QInputDialog>
#include "detectordatabase.h"
#include <QSqlError>
#include <contents.h>
#include <about.h>

using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowTitle("My Project");
    //this->setFixedSize(QSize(570, 300));
    //this->setWindowState(Qt::WindowFullScreen);
    QPixmap pix("C:/Users/ASHRAF/Desktop/X-folder/headingLogo2.png");
    ui->detection_img->setPixmap(pix);
    ui->detection_img->setScaledContents(true);
    ui->detection_img->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);

    ui->tracking_img->setScaledContents(true);
    ui->tracking_img->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);

    ui->Features->addItem("Feature Selection");
    ui->Features->addItem("FAST");
    ui->Features->addItem("SURF");
    ui->Features->addItem("SIFT");
    ui->Features->addItem("ORB");
    ui->Features->addItem("BRIEF");
    ui->spinBox->setEnabled(false);
    ui->horizontalSlider->setEnabled(false);

    addDetector_to_pop_up_manu();
    connect(this, SIGNAL(sendCurrentFrame(int)),ui->spinBox,SLOT(setValue(int)));
    connect(ui->horizontalSlider, SIGNAL(valueChanged(int)), ui->spinBox, SLOT(setValue(int)));
    connect(ui->spinBox, SIGNAL(valueChanged(int)), ui->horizontalSlider, SLOT(setValue(int)));
}

void MainWindow::addDetector_to_pop_up_manu()
{
    ui->Detectors->addItem("Detector Selection");
    Database = QSqlDatabase::addDatabase("QSQLITE");
    Database.setDatabaseName("C:/sqlite2/DetectorList.db");
    Database.open();
    query= new QSqlQuery(Database);
    QString queryString = "select * from DetectorList";
    query->prepare(queryString);

    if(!query->exec())
    {
        QMessageBox::critical(this,tr ("Error"),tr ("Error saving data"));
    }
    else{
        while (query->next()) {
            ui->Detectors->addItem(query->value(0).toString());
        }
    }
    Database.close();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionExit_triggered()
{            
    exit(EXIT_FAILURE);
}

void MainWindow::on_RUN_clicked()
{
    getData_1 =ui->Features->currentText();

    if(getData_1=="Feature Selection" && getData_2=="Detector Selection")
    {
        QMessageBox::warning(this, tr("Warning"),tr("Please select a Feature and a Classfier"));
    }

    if(getData_1=="Feature Selection" && getData_2!="Detector Selection")
    {
        QMessageBox::warning(this, tr("Warning"),tr("Please select a specific Feature detection technique"));
    }

    if(getData_1!="Feature Selection" && getData_2=="Detector Selection")
    {
        QMessageBox::warning(this, tr("Warning"),tr("Please select a specific Detector"));
    }

    if(getData_1!="Feature Selection" && getData_2!="Detector Selection")
    {
        if (image_path_detect.empty() && video_path_track.empty())
        {
            QMessageBox::warning(this, tr("Warning"),tr("Neither Image nor Video has been chosen for object detection. \n Please choose first image or video for object detection"));
        }
        else
        {
            Feature_selection();
        }
    }
}

void MainWindow::Feature_selection()
{        
    if(getData_1=="FAST")
    {
        FAST_corner_detection();
    }
    else if(getData_1=="SIFT")
    {      
        SIFT_feature_detection();
    }
    else if(getData_1=="SURF")
    {
        SURF_feature_detection();
    }
    else if(getData_1=="BRIEF")
    {
        BRIEF_feature_detection();
    }
    else if(getData_1=="ORB")
    {
        ORB_feature_detection();
    }
}

void MainWindow::Classifier_selection()
{
    Database = QSqlDatabase::addDatabase("QSQLITE");
    Database.setDatabaseName("C:/sqlite2/DetectorList.db");
    Database.open();
    query= new QSqlQuery(Database);
    QString queryString = "select * from DetectorList";
    query->prepare(queryString);
    QString path;

    if(!query->exec())
    {
        QMessageBox::critical(this,tr ("Error"),tr ("Error saving data"));
    }
    while (query->next())
    {
        if (getData_2==query->value(0).toString())
        {
            path=query->value(1).toString().toLocal8Bit().constData();
            classifier=path.toLocal8Bit().constData();
            Classifier_Apply();
        }
    }
    Database.close();
}

void MainWindow::Classifier_Apply()
{    
    DisplayImages();
    if(frame.empty())
    {
        QMessageBox::warning(this, tr("Warning"),tr("Detection image has not been selected. Please select a image for particular object detection."));
    }
    else if( !cascade.load( classifier ) ) //-- 1. Load the cascade
    {
        QMessageBox::warning(this, tr("Warning"),tr("Detector cannot be loaded. Please select a specific Detector and add it's path in the Database system."));
    }
    else if( !frame.empty() ) //-- 2. Apply the classifier to the frame
    { //detectAndDisplay( frame );

        Mat frame_gray;
        RNG rng(12345);

        Mat img = frame;
        cvtColor( frame, frame_gray, CV_BGR2GRAY );
        equalizeHist( frame_gray, frame_gray );

        //-- Detect faces
        cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        for( size_t i = 0; i < faces.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            rectangle( frame, faces[i].tl() , faces[i].br(), color, 2, 8, 0 );
        }

        Mat3b src = frame;
        QImage dest=Mat3b2QImage(src);
        ui->detection_img->setPixmap(QPixmap::fromImage(dest));
        ui->detection_img->setScaledContents(true);
        ui->detection_img->setSizePolicy( QSizePolicy::Ignored, QSizePolicy::Ignored );

        QPoint point;
        connect(ui->detection_img,SIGNAL(sendMousePosition(QPoint&)),this,SLOT(showMousePosition(QPoint&)));
        imwrite("object_detected_image.jpg",frame);
    }
}

void MainWindow::DisplayImages()
{
    if(!image_path_detect.empty())
    {
        frame= imread(image_path_detect);
        frame2= imread(image_path_detect);
    }
    if(!image_path_track.empty())
    {
        frame3 =imread(image_path_track);
    }

    if(!video_path_track.empty())
    {
        cap.set(CV_CAP_PROP_POS_FRAMES, CurrentFrame);
        cap >> frame;
        cap.set(CV_CAP_PROP_POS_FRAMES, CurrentFrame);
        cap >>frame2;
        cap.set(CV_CAP_PROP_POS_FRAMES, CurrentFrame+1);
        cap >> frame3;        
    }
}

void MainWindow::FAST_corner_detection()
{
    inputImg_1=object;
    Mat GrayImage_1,GrayImage_2;

    if(inputImg_1.empty())
    {
        Classifier_Apply();
    }

    if(inputImg_1.channels() >2)
    {
        cvtColor( inputImg_1, GrayImage_1, CV_BGR2GRAY ); // converting color to gray image
    }
    else
    {
        GrayImage_1 =  inputImg_1;
    }
    if((!image_path_track.empty()) || (!video_path_track.empty()))
    {
        inputImg_2=frame3;
        if(inputImg_2.channels() >2)
        {
            cvtColor( inputImg_2, GrayImage_2, CV_BGR2GRAY ); // converting color to gray image
        }
        else
        {
            GrayImage_2 =  inputImg_2;
        }

        vector<KeyPoint>keypoints1, keypoints2;

        Ptr<FastFeatureDetector> detector = FastFeatureDetector::create("FAST");
        detector->detect(GrayImage_1, keypoints1, Mat());
        detector->detect(GrayImage_2, keypoints2, Mat());

        SurfDescriptorExtractor extractor;

        Mat descriptors_object, descriptors_scene;

        extractor.compute( GrayImage_1, keypoints1, descriptors_object );
        extractor.compute( GrayImage_2, keypoints2, descriptors_scene );

        BFMatcher matcher(NORM_L2);
        vector<DMatch> matches;
        matcher.match(descriptors_object, descriptors_scene, matches);


        Mat3b img_matches;
        drawMatches(inputImg_1, keypoints1, inputImg_2, keypoints2, matches, img_matches);

        QImage pass=Mat3b2QImage(img_matches);
        disp.DisplayQImage(pass);
        disp.setModal(true);
        disp.exec();

    }
    else
    {
        QMessageBox::warning(this, tr("Warning"),tr("Image for tracking is not selected yet"));
    }
}

void MainWindow::SIFT_feature_detection()
{
    if(!featureData.nFeatures.isEmpty())
        nFeatures=featureData.nFeatures.toDouble();
    else
        nFeatures=0;
    if(!featureData.nOctaveLayers.isEmpty())
        nOctaveLayers=featureData.nOctaveLayers.toDouble();
    else
        nOctaveLayers=4;
    if(!featureData.edgeThreshold.isEmpty())
        edgeThreshold=featureData.edgeThreshold.toDouble();
    else
        edgeThreshold=10;
    if(!featureData.sigma.isEmpty())
        sigma=featureData.sigma.toDouble();
    else
        sigma=1.6;  

    contrastThreshold=0.04;
    Mat inputImg,outputImg;
    inputImg = object;
    if((!image_path_track.empty())||(!video_path_track.empty()))
    {
        outputImg=frame3;
        cvtColor( outputImg, outputImg, CV_BGR2GRAY );        

        // Verify the images loaded successfully.
        if( !inputImg.data || !outputImg.data )
        {
            QMessageBox::warning(this, tr("Warning"),tr("No Image has been chosen for object tracking. \n Please choose first image or video for object tracking"));
            exit(0);
        }

        // Detect keypoints in both images.
        FeatureDetector* detector;
        detector = new SiftFeatureDetector(
                    nFeatures, //0
                    nOctaveLayers, //4
                    contrastThreshold, //0.001
                    edgeThreshold, //2.9
                    sigma //1.6
                    );

        DescriptorExtractor* extractor;
        extractor = new SiftDescriptorExtractor();

        vector<KeyPoint> inputImgKeypoints,outputImgKeypoints;
        detector->detect(inputImg, inputImgKeypoints);
        detector->detect(outputImg, outputImgKeypoints);

        // Print how many keypoints were found in each image.

        Mat inputImgDescriptors, outputImgDescriptors;
        extractor->compute(inputImg, inputImgKeypoints, inputImgDescriptors);
        extractor->compute(outputImg, outputImgKeypoints, outputImgDescriptors);

        // Print some statistics on the matrices returned.
        Size size = inputImgDescriptors.size();

        size = outputImgDescriptors.size();

        BFMatcher matcher(NORM_L2);
        vector<DMatch> matches;
        matcher.match(inputImgDescriptors, outputImgDescriptors, matches);

        // Draw the results. Displays the images side by side, with colored circles at
        // each keypoint, and lines connecting the matching keypoints between the two
        // images.

        Mat3b img_matches;
        drawMatches(object, inputImgKeypoints, frame3, outputImgKeypoints, matches, img_matches);

        QImage pass=Mat3b2QImage(img_matches);
        disp.DisplayQImage(pass);
        disp.setModal(true);
        disp.exec();
    }
    else
    {
        QMessageBox::warning(this, tr("Warning"),tr("Image for tracking is not selected yet"));
    }
}

void MainWindow::SURF_feature_detection()
{
    Mat img_object = object;
    cvtColor(img_object,img_object,cv::COLOR_RGB2GRAY);
    Mat img_scene = frame3;
    cvtColor(img_scene, img_scene, CV_BGR2GRAY);
    if( !img_object.data || !img_scene.data )
    {
        QMessageBox::warning(this, tr("Warning"),tr("No Image has been chosen for object tracking. \n Please choose first image or video for object tracking"));
        exit(0);
    }

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian;// = 5;

    if(!featureData.minHessian.isEmpty())
        minHessian=featureData.minHessian.toDouble();
    else
        minHessian=5;

    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    detector.detect( img_object, keypoints_object );
    detector.detect( img_scene, keypoints_scene );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_object, descriptors_scene;

    extractor.compute( img_object, keypoints_object, descriptors_object );
    extractor.compute( img_scene, keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
        { good_matches.push_back( matches[i]); }
    }

    Mat3b img_matches;
    drawMatches(object, keypoints_object, frame3, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Localize the object
    vector<Point2f> obj;
    vector<Point2f> scene;

    for(unsigned int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    Mat H = findHomography( obj, scene, CV_RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
    vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);


    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

    //-- Show detected matches
    QImage pass=Mat3b2QImage(img_matches);
    disp.DisplayQImage(pass);
    disp.setModal(true);
    disp.exec();

}

void MainWindow::BRIEF_feature_detection()
{
    Mat img_1, grayImg_1, img_2, grayImg_2;
    img_1 = object;
    img_2=frame3;
    if( !img_1.data || !img_2.data )
    {
        QMessageBox::warning(this, tr("Warning"),tr("No Image has been chosen for object tracking. \n Please choose first image or video for object tracking"));
        exit(0);
    }

    if(img_1.channels()>2)
    {
        cvtColor(img_1,grayImg_1, CV_BGR2GRAY);
    }
    else
    {
        grayImg_1=img_1;
    }
    if(img_2.channels()>2)
    {
        cvtColor(img_2,grayImg_2, CV_BGR2GRAY);
    }
    else
    {
        grayImg_2=img_2;
    }
    vector<KeyPoint>keypoints_1, keypoints_2;

    FastFeatureDetector detector;
    detector.detect(grayImg_1,keypoints_1);
    detector.detect(grayImg_2,keypoints_2);

    Mat img1_descriptor, img2_descriptor;
    BriefDescriptorExtractor extractor;
    extractor.compute(grayImg_1,keypoints_1, img1_descriptor );
    extractor.compute(grayImg_2,keypoints_2, img2_descriptor );

    Size size = img1_descriptor.size();

    size = img2_descriptor.size();

    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(img1_descriptor, img2_descriptor, matches);

    Mat3b img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);

    QImage pass=Mat3b2QImage(img_matches);
    disp.DisplayQImage(pass);
    disp.setModal(true);
    disp.exec();
}

void MainWindow::ORB_feature_detection()
{

    Mat img_1, grayImg_1, img_2, grayImg_2;
    Mat img1_descriptor, img2_descriptor;
    img_1 =object;
    img_2=frame3;
    ORB orb;
    int method = 0;
    if( !img_1.data || !img_2.data )
    {
        QMessageBox::warning(this, tr("Warning"),tr("No Image has been chosen for object tracking. \n Please choose first image or video for object tracking"));
        exit(0);
    }

    if(img_1.channels()>2)
    {
        cvtColor(img_1,grayImg_1, CV_BGR2GRAY);
    }
    else
    {
        grayImg_1=img_1;
    }
    if(img_2.channels()>2)
    {
        cvtColor(img_2,grayImg_2, CV_BGR2GRAY);
    }
    else
    {
        grayImg_2=img_2;
    }
    vector<KeyPoint>keypoints_1, keypoints_2;

    OrbFeatureDetector detector(25, 1.0f, 2, 10, 0, 2, 0, 10);
    OrbDescriptorExtractor extractor;
    if( method == 0 ) { //-- ORB
        orb.detect(grayImg_1, keypoints_1);
        orb.detect(grayImg_2, keypoints_2);
        orb.compute(grayImg_1, keypoints_1, img1_descriptor);
        orb.compute(grayImg_2, keypoints_2, img2_descriptor);
    } else { //-- SURF test
        detector.detect(grayImg_1, keypoints_1);
        detector.detect(grayImg_2, keypoints_2);
        extractor.compute(grayImg_1, keypoints_1, img1_descriptor);
        extractor.compute(grayImg_2, keypoints_2, img2_descriptor);
    }

    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(img1_descriptor, img2_descriptor, matches);

    Mat3b img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);

    QImage pass=Mat3b2QImage(img_matches);
    disp.DisplayQImage(pass);
    disp.setModal(true);
    disp.exec();
}

void MainWindow::DisplayDetectedObj()
{
    Point2f point3(point2.x(),point2.y());
    float a,b,c,d,e,f,x,y;
    a=point3.x; b=480; c=frame.cols;
    d=point3.y; e=320; f=frame.rows;

    x =static_cast<int>(a/b*c);
    y =static_cast<int>(d/e*f);

    Point2f pot(x,y);
    for( size_t i = 0; i < faces.size(); i++ )
    {
        Point2f p1,p2,p3,p4;
        p1= Point2f( faces[i].x, faces[i].y);
        p2= Point2f( faces[i].x + faces[i].width, faces[i].y);
        p3= Point2f( faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        p4= Point2f( faces[i].x , faces[i].y + faces[i].height);

        if(pot.x>p1.x && pot.x<p3.x && pot.y>p1.y && pot.y<p3.y)
        {
            object = frame2(faces[i]);
            imwrite("object5.jpg", object);
            namedWindow("object", CV_WINDOW_FREERATIO);
            imshow("object", object);
        }
    }
}

void MainWindow::on_actionDetect_triggered()
{
    video_path_track.clear();
    ui->detection_img->clear();
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Images (*.png *.xpm *.jpg)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString imagefileName = QFileDialog::getOpenFileName(this, tr("Open File"), "C:/Users/ASHRAF/Desktop/", tr("Images (*.png *.xpm *.jpg)"));

    if(!imagefileName.isEmpty())
    {
        QImage image(imagefileName);
        ui->detection_img->setPixmap(QPixmap::fromImage(image));
        image_path_detect=imagefileName.toLocal8Bit().constData();
        ui->spinBox->setEnabled(false);
        ui->horizontalSlider->setEnabled(false);
    }
}

void MainWindow::on_actionVideo_triggered()
{
    image_path_track.clear();
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Videos (*.avi)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString videofileName = QFileDialog::getOpenFileName(this, tr("Open File"), "C:/Users/ASHRAF/Desktop/", tr("Videos (*.avi)"));

    if(!videofileName.isEmpty())
    {
        video_path_track= videofileName.toLocal8Bit().constData();
        cap=VideoCapture(video_path_track);
        if(!cap.isOpened())
        {
            QMessageBox::warning(this, tr("Warning"),tr("Error loadeing video."));
        }
        else
        {
            cap >> frame;            
            if(frame.empty())
            {
                QMessageBox::warning(this, tr("Warning"),tr("Video frame cannot be openned."));
            }
            else
            {
                CurrentFrame=cap.get(CV_CAP_PROP_POS_FRAMES);
                int TotalFrame=cap.get(CV_CAP_PROP_FRAME_COUNT);
                ui->spinBox->setMaximum(TotalFrame);
                ui->horizontalSlider->setMaximum(ui->spinBox->maximum());

                emit sendCurrentFrame(CurrentFrame);

                image1= Mat3b2QImage(frame);
                ui->detection_img->setPixmap(QPixmap::fromImage(image1));
                QApplication::processEvents();

                cap.set(CV_CAP_PROP_POS_FRAMES, CurrentFrame+1);
                cap >> frame3;
                image2= Mat3b2QImage(frame3);
                ui->tracking_img->setPixmap(QPixmap::fromImage(image2));
                QApplication::processEvents();
                cap.set(CV_CAP_PROP_POS_FRAMES, CurrentFrame);

                ui->spinBox->setEnabled(true);
                ui->horizontalSlider->setEnabled(true);
            }
        }
    }
}
void MainWindow::on_actionTrack_triggered()
{
    ui->tracking_img->clear();
    video_path_track.clear();
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Images (*.png *.xpm *.jpg)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString imagefileName = QFileDialog::getOpenFileName(this, tr("Open File"), "C:/Users/ASHRAF/Desktop/", tr("Images (*.png *.xpm *.jpg)"));

    if(!imagefileName.isEmpty())
    {
        QImage image(imagefileName);
        ui->tracking_img->setPixmap(QPixmap::fromImage(image));
        image_path_track=imagefileName.toLocal8Bit().constData();
        ui->spinBox->setEnabled(false);
        ui->horizontalSlider->setEnabled(false);
    }
}

void MainWindow::on_Detectors_currentIndexChanged(const QString &arg1)
{
    getData_2 = arg1;
    Classifier_selection();
}

void MainWindow::showMousePosition(QPoint &pos)
{    
    point2 = pos;
    DisplayDetectedObj();
}

QImage MainWindow::Mat3b2QImage (Mat3b src)
{
    QImage dest(src.cols, src.rows, QImage::Format_ARGB32);
    for (int y = 0; y < src.rows; ++y) {
        const cv::Vec3b *srcrow = src[y];
        QRgb *destrow = (QRgb*)dest.scanLine(y);
        for (int x = 0; x < src.cols; ++x) {
            destrow[x] = qRgba(srcrow[x][2], srcrow[x][1], srcrow[x][0], 255);
        }
    }
    return dest;
}

void MainWindow::on_detector_clicked()
{
    DetectorDatabase object;
    object.setModal(true);
    object.exec();
}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
    CurrentFrame=arg1;
    cap.set(CV_CAP_PROP_POS_FRAMES, CurrentFrame);
    cap >> frame;    
    image1= Mat3b2QImage(frame);
    ui->detection_img->setPixmap(QPixmap::fromImage(image1));
    QApplication::processEvents();

    cap.set(CV_CAP_PROP_POS_FRAMES, CurrentFrame+1);
    cap >> frame3;
    image2= Mat3b2QImage(frame3);
    ui->tracking_img->setPixmap(QPixmap::fromImage(image2));
    QApplication::processEvents();
    cap.set(CV_CAP_PROP_POS_FRAMES, CurrentFrame);
}

void MainWindow::on_actionContents_triggered()
{
    Contents c;
    c.setModal(true);
    c.exec();
}

void MainWindow::on_actionAbout_triggered()
{
    About c;
    c.setModal(true);
    c.exec();
}

void MainWindow::on_pushButton_clicked()
{
    featureData.setModal(true);
    featureData.exec();
}

void MainWindow::on_Quit_clicked()
{
    exit(EXIT_FAILURE);
}
