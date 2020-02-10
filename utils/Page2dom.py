"""Load modules"""
import os, sys
from xml.dom import minidom
import numpy as np

class Page():
    
    def __init__(self, **kwargs):
        self.image_gt_filename = kwargs.get('image_gt_filename')
        self.image_width = 0
        self.image_height = 0
        self.textRegions = []
        self.imageRegions = []
        self.lineDrawingRegions = []
        self.graphicRegions = []
        self.tableRegions = []
        self.chartRegions = []
        self.separatorRegions = []
        self.mathsRegions = []
        self.noiseRegions = []
        self.frameRegions = []
        self.unknownRegions = []
        self.totalRegions = []
        self.xml = None
        
    def getXML(self, **kwargs):
        if self.xml == None:
            try:
                self.xml = minidom.parse(self.image_gt_filename)
            except (FileNotFoundError, IOError):
                print("{} is not found...".format(self.image_gt_filename))
        return self.xml
    
    def getResolution(self):
        self.image_width = int(self.xml.getElementsByTagName('Page')[0].attributes['imageWidth'].value)
        self.image_height = int(self.xml.getElementsByTagName('Page')[0].attributes['imageHeight'].value)
        return self.image_height,self.image_width
    
    def getRegions(self):
        self.textRegions = self.xml.getElementsByTagName('TextRegion')
        self.imageRegions = self.xml.getElementsByTagName('ImageRegion')
        self.lineDrawingRegions = self.xml.getElementsByTagName('LineDrawingRegion')
        self.graphicRegions = self.xml.getElementsByTagName('GraphicRegion')
        self.tableRegions = self.xml.getElementsByTagName('TableRegion')
        self.chartRegions = self.xml.getElementsByTagName('ChartRegion')
        self.separatorRegions = self.xml.getElementsByTagName('SeparatorRegion')
        self.mathsRegions = self.xml.getElementsByTagName('MathsRegion')
        self.noiseRegions = self.xml.getElementsByTagName('NoiseRegion')
        self.frameRegions = self.xml.getElementsByTagName('FrameRegion')
        self.unknownRegions = self.xml.getElementsByTagName('UnknownRegion')
        self.totalRegions = self.textRegions + self.imageRegions + self.lineDrawingRegions + \
                       self.graphicRegions + self.tableRegions + self.chartRegions + \
                       self.separatorRegions + self.mathsRegions + self.noiseRegions + \
                       self.frameRegions + self.unknownRegions
        return self.totalRegions

    def promptBasicInfo(self):
        print("{} \timageWidth".format(self.image_width))
        print("{} \timageHeight\n".format(self.image_height))

        print("{} \ttextRegion(s)".format(len(self.textRegions)))
        print("{} \timageRegion(s)".format(len(self.imageRegions)))
        print("{} \tlineDrawingRegion(s)".format(len(self.lineDrawingRegions)))
        print("{} \tgraphicRegion(s)".format(len(self.graphicRegions)))
        print("{} \ttableRegion(s)".format(len(self.tableRegions)))
        print("{} \tchartRegion(s)".format(len(self.chartRegions)))
        print("{} \tseparatorRegion(s)".format(len(self.separatorRegions)))
        print("{} \tmathsRegion(s)".format(len(self.mathsRegions)))
        print("{} \tnoiseRegion(s)".format(len(self.noiseRegions)))
        print("{} \tframeRegion(s)".format(len(self.frameRegions)))
        print("{} \tunknownRegion(s)\n".format(len(self.unknownRegions)))
        print("{} \ttotalRegions(s)".format(len(self.totalRegions)))
        