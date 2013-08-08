#include "peoplecount.h"
#include "ui_peoplecount.h"
#include <QPixmap>

PeopleCount::PeopleCount(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::PeopleCount)
{
    ui->setupUi(this);
}

PeopleCount::~PeopleCount()
{
    delete ui;
}

void PeopleCount::on_actionOpen_video_triggered()
{

}


void PeopleCount::showImage(const QImage &img)
{
    ui->label->setPixmap(QPixmap::fromImage(img));
}
