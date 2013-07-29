#include "peoplecount.h"
#include "ui_peoplecount.h"

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
