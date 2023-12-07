#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import NavSatFix
from nmea_msgs.msg import Sentence
import datetime


class NavSatFix2NMEA:
    """
    This class is responsible for converting the NavSatFix message to the GPGGA NMEA.
    """

    def __init__(self):
        self.navsatfix_sub = rospy.Subscriber("fix", NavSatFix, self.navsatfixCallback)
        self.nmea_pub = rospy.Publisher("nmea_sentence", Sentence, queue_size=1)
        self.visible_satellites = 8
        self.nmea_sentence_length = 82
        self.checksum_length = 3
        self.nmea = None
        self.freq = rospy.get_param("~freq", 5.0)
        self.use_ros_time = rospy.get_param("~use_ros_time", False)

    def navsatfixCallback(self, msg: NavSatFix) -> None:
        """
        Callback function for the NavSatFix message. This function converts the NavSatFix

        Args:
            msg (NavSatFix): NavSatFix message containing the GPS data.
        """

        # Time
        if self.use_ros_time:
            hours = int(msg.header.stamp.secs / 3600)
            minutes = int((msg.header.stamp.secs - (hours * 3600)) / 60)
            seconds = msg.header.stamp.secs - (hours * 3600) - (minutes * 60)
        else:
            now = datetime.datetime.now()
            hours = now.hour
            minutes = now.minute
            seconds = now.second

        # Latitude
        lat_degs = int(msg.latitude)
        lat_mins = (msg.latitude - lat_degs) * 60.0
        if lat_degs < 0:
            lat_degs *= -1
            lat_mins *= -1
            lat_dir = "S"
        else:
            lat_dir = "N"

        # Longitude
        lon_degs = int(msg.longitude)
        lon_mins = (msg.longitude - lon_degs) * 60.0
        if lon_degs < 0:
            lon_degs *= -1
            lon_mins *= -1
            lon_dir = "W"
        else:
            lon_dir = "E"

        # Altitude
        alt = msg.altitude

        # GPGGA sentence
        gpgga = "$GPGGA,{:02d}{:02d}{:04.3f},{:02d}{:06.4f},{},{:02d}{:07.4f},{},1,{},3.2,{:03.1f},M,,,,".format(
            hours,
            minutes,
            seconds,
            lat_degs,
            lat_mins,
            lat_dir,
            lon_degs,
            lon_mins,
            lon_dir,
            self.visible_satellites,
            alt,
        )

        # Checksum
        checksum = 0
        for i in range(1, len(gpgga)):
            checksum ^= ord(gpgga[i])
        gpgga += "*{:02X}".format(checksum)

        # Publish the GPGGA sentence
        self.nmea = Sentence()
        self.nmea.header = msg.header
        self.nmea.sentence = gpgga

    def run(self) -> None:
        rate = rospy.Rate(self.freq)
        while not rospy.is_shutdown():
            if self.nmea is not None:
                self.nmea_pub.publish(self.nmea)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("waypoint_planner", anonymous=True)
    N2M = NavSatFix2NMEA()
    N2M.run()
