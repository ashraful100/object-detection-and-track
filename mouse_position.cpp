#include "mouse_position.h"

mouse_position::mouse_position(QWidget *parent): QLabel (parent)
{
 //   this->setMouseTracking(true);

}

//void mouse_position::mouseMoveEvent(QMouseEvent *mouse_event)
//{
//    QPoint pos = mouse_event->pos();
//    if (pos.x() <= this->size().width() && pos.y() <= this->size().height())
//    {
//        if (pos.x() >= 0 && pos.y() >= 0)
//        {
//            emit sendMousePosition(pos);
//        }
//    }

//}

void mouse_position::mousePressEvent(QMouseEvent *mouse_event)
{
    QPoint pos = mouse_event->pos();
    if(mouse_event->button()== Qt::LeftButton || mouse_event->button()== Qt::RightButton)
    {
        emit sendMousePosition(pos);
    }
}
