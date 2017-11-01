#include "featuredata.h"
#include "ui_featuredata.h"

FeatureData::FeatureData(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FeatureData)
{
    ui->setupUi(this);
}

FeatureData::~FeatureData()
{
    delete ui;
}

void FeatureData::on_Save_clicked()
{
    nFeatures= ui->doubleSpinBox->text().replace(",",".");
    nOctaveLayers= ui->doubleSpinBox_2->text().replace(",",".");
    edgeThreshold= ui->doubleSpinBox_3->text().replace(",",".");
    sigma= ui->doubleSpinBox_4->text().replace(",",".");
    minHessian= ui->spinBox->text().replace(",",".");


    close();
}

void FeatureData::on_Cancel_clicked()
{
    nFeatures.clear();
    nOctaveLayers.clear();
    edgeThreshold.clear();
    sigma.clear();
    minHessian.clear();

    close();
}
