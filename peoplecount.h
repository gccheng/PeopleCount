#ifndef PEOPLECOUNT_H
#define PEOPLECOUNT_H

#include <QMainWindow>
#include <QImage>

namespace Ui {
class PeopleCount;
}

class PeopleCount : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit PeopleCount(QWidget *parent = 0);
    ~PeopleCount();

public slots:
    void showImage(const QImage &img);

private slots:
    void on_actionOpen_video_triggered();

private:
    Ui::PeopleCount *ui;
};

#endif // PEOPLECOUNT_H
