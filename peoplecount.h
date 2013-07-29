#ifndef PEOPLECOUNT_H
#define PEOPLECOUNT_H

#include <QMainWindow>

namespace Ui {
class PeopleCount;
}

class PeopleCount : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit PeopleCount(QWidget *parent = 0);
    ~PeopleCount();
    
private:
    Ui::PeopleCount *ui;
};

#endif // PEOPLECOUNT_H
