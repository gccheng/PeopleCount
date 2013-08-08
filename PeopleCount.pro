#-------------------------------------------------
#
# Project created by QtCreator 2013-05-17T16:33:49
#
#-------------------------------------------------

QT       += core gui

TARGET = PeopleCount
TEMPLATE = app


SOURCES += main.cpp\
        peoplecount.cpp \
    digitrecognizer.cpp \
    counter.cpp

INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_ml -lopencv_video -lopencv_objdetect

HEADERS  += peoplecount.h \
    histograms.h \
    digitrecognizer.h \
    counter.h \
    configure.h

FORMS    += peoplecount.ui

OTHER_FILES += \
    memo.txt \
    oldcode.txt
