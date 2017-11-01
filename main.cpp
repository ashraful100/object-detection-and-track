#include "mainwindow.h"
#include <QApplication>
#include <iostream>
#include <QPushButton>

using namespace std;

int main(int argc, char *argv[])
{   
    QApplication a(argc, argv);
    //QPushButton *button = new QPushButton ("Add the numbers");
    //QObject::connect(button, SIGNAL(), &app, SLOT());
    MainWindow w;

    w.show();

    return a.exec();
}
