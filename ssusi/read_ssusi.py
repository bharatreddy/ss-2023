import netCDF4
# from cdf.internal import EPOCHbreakdown
import pandas
import os
import numpy
import datetime
import matplotlib.pyplot as plt
import aacgmv2

class ProcessData(object):
    """
    A class to Download SSUSI data
    given a date and datatype!
    """
    def __init__(self, inpDirs, outDir, inpDate, dataType="sdr"):
        """
        Given a list of dirs (SSUSI has multiple files per date
        for the same satellite). Read all files from it.
        """
        # loop through the dir and get a list of the files
        self.fileList = []
        for currDir in inpDirs:
            for root, dirs, files in os.walk(currDir):
                for fNum, fName in enumerate(files):
                    if dataType.upper() in fName:
                        self.fileList.append( root + fName )
        self.outDir = outDir
        self.inpDate = inpDate

    def processed_data_to_file(self, coords="geo", keepRawFiles=False):
        """
        read the required data into a dataframe
        select only required columns, if aacgm
        coordinates are selected convert geo to
        AACGM coords and save data to file!
        """
        from functools import reduce
        # We'll delete raw File dirs at the end, keep alist of them
        delDirList = []
        for fileInd, currFile in enumerate(self.fileList):
            # if selFname not in currFile:
            #     continue
            # Get Sat name
            print("currently working with file-->", currFile)
            print("processing--->", fileInd+1, "/", len(self.fileList), "files")
            satName = "F18"
            if "F17" in currFile:
                satName = "F17"
            if "F16" in currFile:
                satName = "F16"
            if ( (".nc" not in currFile) & (".NC" not in currFile) ):
                print("Not a valid netcdf file!!")
                continue
            currDataSet = netCDF4.Dataset(currFile)

            dtList = self.cdf_epoch_to_datetime(currDataSet.variables["TIME_EPOCH_DAY"][:])
            currDate = self.inpDate.strftime("%Y%m%d")
            # get peircepoints
            prpntLats = currDataSet.variables['PIERCEPOINT_DAY_LATITUDE'][:]
            prpntLons = currDataSet.variables['PIERCEPOINT_DAY_LONGITUDE'][:]
            prpntAlts = currDataSet.variables['PIERCEPOINT_DAY_ALTITUDE'][:]
            # GET DISK intensity data - waveband/color radiance data
            # 5 colors are - 121.6, 130.4, 135.6 nm and LBH short and LBH long
            dskInt121 = currDataSet.variables['DISK_INTENSITY_DAY'][:, :, 0]
            dskInt130 = currDataSet.variables['DISK_INTENSITY_DAY'][:, :, 1]
            dskInt135 = currDataSet.variables['DISK_INTENSITY_DAY'][:, :, 2]
            dskIntLBHS = currDataSet.variables['DISK_INTENSITY_DAY'][:, :, 3]
            dskIntLBHL = currDataSet.variables['DISK_INTENSITY_DAY'][:, :, 4]
            # We'll store the data in a DF. Now we need to be a little cautious
            # when storing the data in a DF. SSUSI measures flux as swaths, so
            # at each time instance we have multiple lats and lons and disk data.
            # I'm taking a simple approach where I take each lat (lon and other
            #  data) at a time instance as a column and time as rows. So if 
            # the array ishaving a dimention of 42x1632, each of the 42 
            # elements becomes a column and the 1632 time instances become rows.
            latColList = [ "glat." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
            lonColList = [ "glon." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
            d121ColList = [ "d121." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
            d130ColList = [ "d130." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
            d135ColList = [ "d135." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
            dLBHSColList = [ "dlbhs." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
            dLBHLColList = [ "dlbhl." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
            # create dataframes with
            dfLat = pandas.DataFrame(prpntLats.T,columns=latColList, index=dtList)
            dfLon = pandas.DataFrame(prpntLons.T,columns=lonColList, index=dtList)
            dfD121 = pandas.DataFrame(dskInt121.T,columns=d121ColList, index=dtList)
            dfD130 = pandas.DataFrame(dskInt130.T,columns=d130ColList, index=dtList)
            dfD135 = pandas.DataFrame(dskInt135.T,columns=d135ColList, index=dtList)
            dfDLBHS = pandas.DataFrame(dskIntLBHS.T,columns=dLBHSColList, index=dtList)
            dfDLBHL = pandas.DataFrame(dskIntLBHL.T,columns=dLBHLColList, index=dtList)
            # Merge the dataframes
            ssusiDF = reduce(lambda left,right: pandas.merge(left,right,\
                         left_index=True, right_index=True), [ dfLat, \
                        dfLon, dfD121, dfD130, dfD135, dfDLBHL, dfDLBHS ])
            ssusiDF["orbitNum"] = currDataSet.variables['ORBIT_DAY'][:]
            # Lets also keep track of the sat name and shape of arrays
            ssusiDF["sat"] = satName
            ssusiDF["shapeArr"] = prpntLats.shape[0]
            # # reset index, we need datetime as a col
            ssusiDF = ssusiDF.reset_index()
            ssusiDF = ssusiDF.rename(columns = {'index':'date'})
            if coords != "geo":
                # Now we need to convert the GLAT, GLON into MLAT, MLON and MLT
                ssusiDF = ssusiDF.apply(self.convert_to_aacgm, axis=1)
                ssusiDF = ssusiDF.round(2)
                # We'll only need aacgm coords, discard all geo coords
                mlatColList = [ "mlat." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
                mlonColList = [ "mlon." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
                mltColList = [ "mlt." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
                outCols = ["date", "sat", "orbitNum"] + mlatColList + mlonColList + mltColList + d121ColList + \
                            d130ColList + d135ColList + dLBHSColList + dLBHLColList
            else:
                outCols = ["date", "sat", "orbitNum", "shapeArr"] + latColList + lonColList + d121ColList + \
                            d130ColList + d135ColList + dLBHSColList + dLBHLColList
            ssusiDF = ssusiDF[ outCols ]
            # We now need to write the processed data to a file
#             if not os.path.exists(self.outDir + "/" +satName):
#                 os.makedirs(self.outDir + "/" + satName)
            # if the file for the date exists append data
            # else create the file and write data!!!
            outFileDir = self.outDir + "/" + satName + "_prcsd" + "/"
            if not os.path.exists(outFileDir):
                os.makedirs(outFileDir)
            outFileName = outFileDir + currDate + ".txt"
            if not os.path.exists( outFileName ):
                # NOTE we only need header when writing data for the first time!
                with open(outFileName, 'w') as ftB:
                    ssusiDF.to_csv(ftB, header=True,\
                                      index=False, sep=' ' )
            else:
                # sometimes the file is already present!
                # from a previous run.
                # we'll simply append to existing data
                # in that case!. So delete the existing file
                # if that is the case.
                if fileInd == 0:
                    print("FILE Exists already! deleting and overwriting")
                    os.remove(outFileName)
                    with open(outFileName, 'w') as ftB:
                        ssusiDF.to_csv(ftB, header=True,\
                                          index=False, sep=' ' )
                else:
                    with open(outFileName, 'a') as ftB:
                        ssusiDF.to_csv(ftB, header=False,\
                                          index=False, sep=' ' )
            if not keepRawFiles:
                os.remove(currFile)
                currDelDir = "/".join(currFile.split("/")[:-1]) + "/"
                if currDelDir not in delDirList:
                    delDirList.append( currDelDir )
        if not keepRawFiles:
            for dd in delDirList:
                os.rmdir(dd)


    def convert_to_aacgm(self, row):
        """
        For the SSUSI DF convert all the 42
        Given glat, glon and date return
        mlat, mlon and mlt
        """
        for i in range( row["shapeArr"] ):
            indStr = str(i+1)
            mlat, mlon = aacgmv2.convert(row["glat." + indStr], row["glon." + indStr],\
                               300, row["date"])
            # mlon, mlat = utils.coord_conv( row["glon." + indStr], row["glat." + indStr], \
            #                      "geo", "mag", altitude=300., \
            #                      date_time=row["date"] )
            mlt = aacgmv2.convert_mlt(mlon, row["date"], m2a=False)
            row["mlat." + indStr] = numpy.round( mlat, 2)
            row["mlon." + indStr] = numpy.round( mlon, 2)
            row["mlt." + indStr] = numpy.round( mlt, 2)
        return row
    
    
    def breakdown_epoch(self, epochs, to_np=True): 
        """
        Copied code from:
        https://cdflib.readthedocs.io/en/latest/_modules/cdflib/epochs.html
        """

        if (isinstance(epochs, float) or isinstance(epochs, numpy.float64)):
            new_epochs = [epochs]
        elif (isinstance(epochs, list) or isinstance(epochs, tuple) or
              isinstance(epochs, numpy.ndarray)):
            new_epochs = epochs
        else:
            raise ValueError('Bad data')
        count = len(new_epochs)
        components = []
        for x in range(0, count):
            component = []
            epoch = new_epochs[x]
            if (epoch == -1.0E31):
                component.append(9999)
                component.append(12)
                component.append(31)
                component.append(23)
                component.append(59)
                component.append(59)
                component.append(999)
            else:
                if (epoch < 0.0):
                    epoch = -epochs
                if (isinstance(epochs, int)):
                    epoch = float(epoch)
                msec_AD = epoch
                second_AD = msec_AD / 1000.0
                minute_AD = second_AD / 60.0
                hour_AD = minute_AD / 60.0
                day_AD = hour_AD / 24.0
                jd = int(1721060 + day_AD)
                l = jd+68569
                n = int(4*l/146097)
                l = l-int((146097*n+3)/4)
                i = int(4000*(l+1)/1461001)
                l = l-int(1461*i/4)+31
                j = int(80*l/2447)
                k = l-int(2447*j/80)
                l = int(j/11)
                j = j+2-12*l
                i = 100*(n-49)+i+l
                component.append(i)
                component.append(j)
                component.append(k)
                component.append(int(hour_AD % 24.0))
                component.append(int(minute_AD % 60.0))
                component.append(int(second_AD % 60.0))
                component.append(int(msec_AD % 1000.0))
            if (count == 1):
                if to_np:
                    return numpy.array(component)
                else:
                    return component
            else:
                components.append(component)
        if to_np:
            return numpy.array(components)
        else:
            return components
        
    def cdf_epoch_to_datetime(self, cdf_time, to_np=True):
        """
        Encodes the epoch(s) into Python datetime.  
        Copied code from:
        https://cdflib.readthedocs.io/en/latest/_modules/cdflib/epochs.html
        """
        time_list = self.breakdown_epoch(cdf_time, to_np=True)
        if isinstance(time_list[0], (float, int, complex)):  # single time
            time_list = [time_list]

        if len(time_list[0]) >= 8:
            dt = [datetime.datetime(t[0], t[1], t[2], t[3], t[4], t[5], microsecond=t[6]*1000+t[7]) for t in time_list]
        elif len(time_list[0]) == 7:
            dt = [datetime.datetime(t[0], t[1], t[2], t[3], t[4], t[5], microsecond=t[6]*1000) for t in time_list]
        elif len(time_list[0]) == 6:
            dt = [datetime.datetime(t[0], t[1], t[2], t[3], t[4], t[5]) for t in time_list]
        else:
            raise ValueError('unknown cdf_time format')

        return numpy.array(dt) if to_np else dt

