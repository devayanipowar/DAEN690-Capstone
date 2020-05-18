import mgrs
import warnings
import shapely
from shapely.geometry import  Point, Polygon, MultiPolygon
import string

m = mgrs.MGRS()


def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

def getnextletter(letter,easting,main, s="none"):
    
    otherletters = None
    if easting == True:
        #replace columns (A–Z, omitting I and O)
        latitude, _ = m.toLatLon(s.encode('utf-8'))
        # print(latitude)
        if latitude < 25.962984: #RS HJ ZA do not exist above this latitude...
            letters = string.ascii_uppercase.replace('I','').replace('O','')
            troubleletters = ['R','H','Z']
        else:
            letters = string.ascii_uppercase.replace('I','').replace('O','').replace('H','').replace('J','').replace('R','').replace('S','').replace('Z','').replace('A','')
            troubleletters = ['Q','G','Y']
    else:
        #northing and replace row (A–V, omitting I and O)
        letters = string.ascii_uppercase.replace('I','').replace('O','')[:20]
    letterpos = letters.find(letter)


    if easting and (main[0] in troubleletters): # if easting with first main character as RHorZ if below latline or QGY if above
        issues = True       #east -- but the second letter also changes based on random flips
        print(s)
        print('ATTENTION! EASTING FLIP MAY NOT BE CORRECT')
        #if this is an issue we need to find the second letter of the cell to the right
        (lat,lon) = m.toLatLon(s.encode('utf-8')) #lat/lon of current cell
        if len(s) >= 6: #if not 100km
            (lat,lon) = m.toLatLon(s[:5].encode('utf-8')) #get back to SQ corner of 100k
        # print((lat,lon))
        easterngrid = m.toMGRS(lat+.5, lon+3) #move to the esat and get a cell
        # print(easterngrid)
        #get the right grid letter
        otherletters = []
        otherletters.append(easterngrid.decode()[4])
        #we need the next of the east square's second character for the NE corner
        northletters = string.ascii_uppercase.replace('I','').replace('O','')[:20]
        letterpostw = northletters.find(easterngrid.decode()[4])
        if letterpostw+1 != len(northletters):
            newnorthingletter = northletters[letterpostw+1]
        else:
            newnorthingletter = northletters[0] 
        
        #also the zone may change
        otherletters.append(newnorthingletter)
        otherletters.append(easterngrid.decode()[:3])



    #for all other cases use the next letter in respective list or loop to start
    if letterpos+1 != len(letters):
        letter = letters[letterpos+1]
    else:
        letter = letters[0] 
    # print(letter)
    return letter, otherletters

def getlatlons(s):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    m = mgrs.MGRS()
    # print(s)
    otherletters = None
    prematureeastingflip = False
    prematurezonestartflip = False
    if len(s) >= 6: #if this is not 100k 

        #establish params
        main = s[3:5]
        coords = s[5:]
        zone = s[:3]
        coordlength = int(len(coords)/2) #adjust for precision

        #for rounding tricks
        mainest = main 
        mainnth = main
        mainboth = main
        eastzone = zone

        #assign easting and northing and +1 values
        easting = str(coords[:coordlength])
        northing = str(coords[coordlength:])
        eastingadded = str(int(easting)+1)
        northingadded = str(int(northing)+1)

        #issues  with +1 rounding if easting or northing is "00000#"
        if len(eastingadded)< len(easting):
            eastingadded = str((len(easting)-len(eastingadded))*'0'+eastingadded)
        if len(northingadded)< len(northing):
            northingadded = str((len(northing)-len(northingadded))*'0'+northingadded)


        #zone flipping rules        

        #flipping when adding +1 doesn't work, we need to move to the next column or row letter
        #easting
        if str(coords[:coordlength]) == "9"*coordlength: #if 9999 in easting or premature flip on utm zone boundary
            eastingadded = len(str(easting))*'0' 
            newletter, otherletters  =  getnextletter(main[0],True, main, s)
            temp = list(main)
            temp[0] = newletter
            mainest = "".join(temp)
            if otherletters != None:
                mainest = newletter + otherletters[0]

        #northing
        if str(coords[coordlength:]) == "9"*coordlength: # if 9999 in northing
            northingadded = len(str(northing))*'0'
            newletter, _ =  getnextletter(main[1],False, main)
            temp = list(main)
            temp[1] = newletter
            mainnth = "".join(temp)

        #general rounding if flip occurs
        mainboth = mainest[0] + mainnth[1]
        if otherletters != None:
            mainboth = mainest[0] + otherletters[1]
            eastzone = otherletters[2]


        # check if special flips occur if in troubleletter 100km zone (flip main zone before 99999)
        if main[0] in ['H', 'Z','R','Q','G','Y']:
            #get next zone list
            latitude, longitude = m.toLatLon(s.encode('utf-8'))
            # print(latitude)
            if latitude < 25.962984: #RS HJ ZA do not exist above this latitude...
                letters = string.ascii_uppercase.replace('I','').replace('O','')
                troubleletters = ['R','H','Z']
            else:
                letters = string.ascii_uppercase.replace('I','').replace('O','').replace('H','').replace('J','').replace('R','').replace('S','').replace('Z','').replace('A','')
                troubleletters = ['Q','G','Y']
            
            if main[0] in troubleletters:
                #check if eastern UTM zone flip
                se = eastzone + mainest + eastingadded + northing
                fliplat, fliplon = m.toLatLon(se.encode('utf-8'))
                flipzone = utm.from_latlon(fliplat, fliplon)[2]
                if int(zone[:2]) != int(flipzone):
                    print('ATTENTION! Premature easting flip!!')
                    prematureeastingflip = True


        if prematureeastingflip:
            zonelongitudes = [*range(-48,-150,-6)]  #pick nearest lon for eastern boundary
            longitude = closest(zonelongitudes,longitude)
            #easting
            #easting switch overs occur not at 0 but at some value around 7XXX and starts at that  - 10XX -- will add to latitude to correct
            # print(easting)
            eastingadded = str(int('1'+coordlength*'0')-int(easting)-1) #this is not precise enough for X XX or XXX precision
            # print(eastingadded)
            newletter, otherletters  =  getnextletter(main[0],True, main, s)
            temp = list(main)
            temp[0] = newletter
            mainest = "".join(temp)
            if otherletters != None:
                mainest = newletter + otherletters[0]
 

            #northing
            if str(coords[coordlength:]) == "9"*coordlength: # if 9999 in northing
                northingadded = len(str(northing))*'0'
                newletter, _ =  getnextletter(main[1],False, main)
                temp = list(main)
                temp[1] = newletter
                mainnth = "".join(temp)


            #general rounding if flip occurs
            mainboth = mainest
            if str(coords[coordlength:]) == "9"*coordlength: # if 9999 in northing
                mainboth = mainest[0] + otherletters[1]
            eastzone = otherletters[2]
                    
        #also check starting letters of new zone -- b/c mgrs makes things up for the starting position of new zones SW corner lat/lon
        if main[0] in ['J', 'A','S','T','K','B']:
                    #get next zone list
                    latitude, longitude = m.toLatLon(s.encode('utf-8'))
                    # print(latitude)
                    if latitude < 25.962984: #RS HJ ZA do not exist above this latitude...
                        letters = string.ascii_uppercase.replace('I','').replace('O','')
                        troubleletters = ['S','J','A']
                    else:
                        letters = string.ascii_uppercase.replace('I','').replace('O','').replace('H','').replace('J','').replace('R','').replace('S','').replace('Z','').replace('A','')
                        troubleletters = ['T','K','B']
                    
                    if main[0] in troubleletters:
                        #check if eastern UTM zone flip
                        fliplat, fliplon = m.toLatLon(s.encode('utf-8'))
                        flipzone = utm.from_latlon(fliplat, fliplon)[2]
                        if int(zone[:2]) != int(flipzone):
                            print('ATTENTION! Premature zone start flip!!')
                            prematurezonestartflip = True
                            zonelongitudes = [*range(-48,-150,-6)]  #pick nearest lon for eastern boundary
                            longitude = closest(zonelongitudes,longitude)


        #get latlons and output
        southwest = s
        northwest  = zone + mainnth + easting + northingadded
        northeast = eastzone + mainboth + eastingadded + northingadded
        southeast = eastzone + mainest + eastingadded + northing
        # print('NEW')
        # print(s)
        # print(zone)
        # print(main)
        # print(coords)
        # print(southwest)
        # print(northwest)
        # print(northeast)
        # print(southeast)
        # print(m.toLatLon(southwest.encode('utf-8')))
        # print(m.toLatLon(northwest.encode('utf-8')))
        # print(m.toLatLon(northeast.encode('utf-8')))
        # print(m.toLatLon(southeast.encode('utf-8')))
        if not prematureeastingflip and not prematurezonestartflip:
            latlons = [Point(m.toLatLon(southwest.encode('utf-8'))),
                        Point(m.toLatLon(northwest.encode('utf-8'))),
                        Point(m.toLatLon(northeast.encode('utf-8'))),
                        Point(m.toLatLon(southeast.encode('utf-8')))]
        elif prematureeastingflip:
            latlons = [Point(m.toLatLon(southwest.encode('utf-8'))),
                Point(m.toLatLon(northwest.encode('utf-8'))),
                Point(m.toLatLon(northeast.encode('utf-8'))[0] +.0127*10**-(coordlength),longitude),
                Point(m.toLatLon(southeast.encode('utf-8'))[0] +.0127*10**-(coordlength),longitude)] #eastern boundary is fixed 

        elif prematurezonestartflip:
                latlons = [Point(m.toLatLon(southwest.encode('utf-8'))[0]+.0127*10**-(coordlength), longitude),
                    Point(m.toLatLon(northwest.encode('utf-8'))[0]+.0127*10**-(coordlength), longitude),
                    Point(m.toLatLon(northeast.encode('utf-8'))),
                    Point(m.toLatLon(southeast.encode('utf-8')))] #eastern boundary is fixed 
        
        poly = Polygon([(p.y, p.x) for p in latlons]) #lon, lat
        out = [s,poly]
    else:
        #metersquare indent has differnt rules
        main = s[3:]
        zone = s[:3]
        eastzone = zone #rounding

        #north
        newletter, _ =  getnextletter(main[1],False, main)
        temp = list(main)
        temp[1] = newletter
        mainnth = "".join(temp)

        #east
        newletter, otherletters  =  getnextletter(main[0],True, main, s)
        temp = list(main)
        temp[0] = newletter
        mainest = "".join(temp)
        if otherletters != None:
            mainest = newletter + otherletters[0]

        #both 
        mainboth = mainest[0] + mainnth[1]
        if otherletters != None:
            mainboth = mainest[0] + otherletters[1]
            eastzone = otherletters[2]

        southwest = s
        northwest  = zone + mainnth
        northeast = eastzone + mainboth 
        southeast = eastzone + mainest 



        #check if zone change:

        if main[0] in ['H', 'Z','R','Q','G','Y']:
            #get next zone list
            latitude, longitude = m.toLatLon(s.encode('utf-8'))
            # print(latitude)
            if latitude < 25.962984: #RS HJ ZA do not exist above this latitude...
                letters = string.ascii_uppercase.replace('I','').replace('O','')
                troubleletters = ['R','H','Z']
            else:
                letters = string.ascii_uppercase.replace('I','').replace('O','').replace('H','').replace('J','').replace('R','').replace('S','').replace('Z','').replace('A','')
                troubleletters = ['Q','G','Y']
            
            if main[0] in troubleletters:
                #check if eastern UTM zone flip
                se = eastzone + mainest 
                if int(zone[:2]) != int(se[:2]):
                    print('ATTENTION! Premature easting flip!!')
                    prematureeastingflip = True
                    zonelongitudes = [*range(-48,-150,-6)]  #pick nearest lon for eastern boundary
                    longitude = closest(zonelongitudes,longitude)

        #also check starting letters of new zone -- b/c mgrs makes things up for the starting position of new zones SW corner lat/lon
        if main[0] in ['J', 'A','S','T','K','B']:
                    #get next zone list
                    latitude, longitude = m.toLatLon(s.encode('utf-8'))
                    # print(latitude)
                    if latitude < 25.962984: #RS HJ ZA do not exist above this latitude...
                        letters = string.ascii_uppercase.replace('I','').replace('O','')
                        troubleletters = ['S','J','A']
                    else:
                        letters = string.ascii_uppercase.replace('I','').replace('O','').replace('H','').replace('J','').replace('R','').replace('S','').replace('Z','').replace('A','')
                        troubleletters = ['T','K','B']
                    
                    if main[0] in troubleletters:
                        #check if eastern UTM zone flip
                        fliplat, fliplon = m.toLatLon(s.encode('utf-8'))
                        flipzone = utm.from_latlon(fliplat, fliplon)[2]
                        if int(zone[:2]) != int(flipzone):
                            print('ATTENTION! Premature zone start flip!!')
                            prematurezonestartflip = True
                            zonelongitudes = [*range(-48,-150,-6)]  #pick nearest lon for eastern boundary
                            longitude = closest(zonelongitudes,longitude)
        # print('NEW')
        # print(s)
        # print(zone)
        # print(main)
        # print(southwest)
        # print(northwest)
        # print(northeast)
        # print(southeast)
        # print(m.toLatLon(southwest.encode('utf-8')))
        # print(m.toLatLon(northwest.encode('utf-8')))
        # print(m.toLatLon(northeast.encode('utf-8')))
        # print(m.toLatLon(southeast.encode('utf-8')))

        if not prematureeastingflip and not prematurezonestartflip:
            latlons = [Point(m.toLatLon(southwest.encode('utf-8'))),
                        Point(m.toLatLon(northwest.encode('utf-8'))),
                        Point(m.toLatLon(northeast.encode('utf-8'))),
                        Point(m.toLatLon(southeast.encode('utf-8')))]
        elif prematureeastingflip:
            latlons = [Point(m.toLatLon(southwest.encode('utf-8'))),
                Point(m.toLatLon(northwest.encode('utf-8'))),
                Point(m.toLatLon(northeast.encode('utf-8'))[0],longitude),
                Point(m.toLatLon(southeast.encode('utf-8'))[0],longitude)] #eastern boundary is fixed 

        elif prematurezonestartflip:
            latlons = [Point(m.toLatLon(southwest.encode('utf-8'))[0], longitude),
                Point(m.toLatLon(northwest.encode('utf-8'))[0], longitude),
                Point(m.toLatLon(northeast.encode('utf-8'))),
                Point(m.toLatLon(southeast.encode('utf-8')))] #eastern boundary is fixed 

        poly = Polygon([(p.y, p.x) for p in latlons]) #lon, lat
        out = [s,poly]
        # print(poly)
    return out



def buildingcostfunc(row, scale):
    maxx = scale[row['length']][1]
    if row['cost'] == 0:
        return 'one'
    if  (row['cost'] > 0) & (row['cost'] <= maxx/3):
        return 'two'
    if  (row['cost'] > maxx/3) & (row['cost'] <= maxx*2/3):
        return 'three'
    if  (row['cost'] > maxx*2/3) & (row['cost'] <= maxx):
        return 'four'
    return 'five'



if __name__ == "__main__":
    print('no')