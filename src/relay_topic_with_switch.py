#!/usr/bin/env python3

import rospy

from std_msgs.msg import Bool


class RelaySwitch:
    """
    This class is responsible for relaying the messages from relay_in_topic to
    relay_out_topic if the switch is True."""

    def __init__(self):
        self.relay_in_topic = rospy.get_param("~relay_in_topic", "/relay_in")
        self.relay_out_topic = rospy.get_param("~relay_out_topic", "/relay_out")
        self.relay_default_state = rospy.get_param("~relay_default_state", True)

        self.switch_sub = rospy.Subscriber(
            "/switch", Bool, self.switch_callback, queue_size=1
        )
        self.switch = self.relay_default_state

        # Wait for relay_in_topic to be published
        self.waitForInputTopic()
        self.topic_type_ = self.loadTopicType()

        # Dynamically create relay_in_sub and relay_out_pub based on the inferred topic type
        self.relay_in_sub = rospy.Subscriber(
            self.relay_in_topic, self.topic_type_, self.relay_in_callback, queue_size=1
        )
        self.relay_out_pub = rospy.Publisher(
            self.relay_out_topic, self.topic_type_, queue_size=1
        )

    def waitForInputTopic(self) -> None:
        """
        Wait for relay_in_topic to be published
        """

        topic_found = False
        rate = rospy.Rate(0.2)
        while (not topic_found) and (not rospy.is_shutdown()):
            rospy.loginfo("Waiting for topic %s", self.relay_in_topic)
            topics = rospy.get_published_topics()
            for topic in topics:
                if topic[0] == self.relay_in_topic:
                    topic_found = True
                    self.topic_type_ = topic[1]
                    rospy.loginfo("Topic %s found", self.relay_in_topic)
            rate.sleep()

    def loadTopicType(self) -> type:
        """
        Load the topic type from the topic name.
        """

        pkg, msg = self.topic_type_.split("/")
        module = __import__(pkg + ".msg")
        return getattr(module.msg, msg)

    def switch_callback(self, msg: Bool) -> None:
        """
        Callback for the switch topic.

        Args:
            msg (Bool): message from the switch topic.
        """

        self.switch = msg.data

    def relay_in_callback(self, msg: type) -> None:
        """
        Callback to relay the input topic if the switch is enabled.

        Args:
            msg (type): Message from the relay_in_topic.
        """

        if self.switch:
            self.relay_out_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("relay_switch")
    relay_switch = RelaySwitch()
    rospy.spin()
