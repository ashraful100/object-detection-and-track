#include "about.h"
#include "ui_about.h"
#include <QFile>
#include<QMessageBox>
#include<QTextStream>

About::About(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::About)
{
    ui->setupUi(this);

    this->setWindowTitle("About");

    QFile file("C:/Users/ASHRAF/Documents/my_project/About.txt");
    if(!file.open(QIODevice::ReadOnly))
    {
        QMessageBox::information(0,"Info",file.errorString());
    }
    QTextStream In(&file);
    ui->textBrowser->setText(In.readAll());
}

About::~About()
{
    delete ui;
}
