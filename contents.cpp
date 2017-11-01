#include "contents.h"
#include "ui_contents.h"
#include <QFile>
#include<QMessageBox>
#include<QTextStream>

Contents::Contents(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Contents)
{
    ui->setupUi(this);

    this->setWindowTitle("Contents");

    QFile file("C:/Users/ASHRAF/Documents/my_project/Contents.txt");
    if(!file.open(QIODevice::ReadOnly))
    {
        QMessageBox::information(0,"Info",file.errorString());
    }
    QTextStream In(&file);
    ui->textBrowser->setText(In.readAll());
}

Contents::~Contents()
{
    delete ui;
}
