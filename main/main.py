import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gio
import sys
import Home
import livecapture
import imagecapture
import videocapture


class ObjectDetectionExperimental(Gtk.ApplicationWindow):
    def __init__(self, app):
        Gtk.ApplicationWindow.__init__(self, application=app)
        self.set_default_size(800, 600)

        header = Gtk.HeaderBar()
        title_label = Gtk.Label.new("Deep Action Experimental")
        header.set_title_widget(title_label)
        header.set_show_title_buttons(True)
        self.set_titlebar(header)

        menu = Gio.Menu.new()
        menu.append("Home", "app.Home")
        menu.append("Image Capture", "app.imagecapture")
        menu.append("Live Capture", "app.livecapture")
        menu.append("Video Capture", "app.videocapture")
        menu.append("Quit", "app.quit")

        popover = Gtk.PopoverMenu.new_from_model(menu)
        hamburger = Gtk.MenuButton.new()
        hamburger.set_popover(popover)
        hamburger.set_icon_name("open-menu-symbolic")
        header.pack_start(hamburger)

        home_action = Gio.SimpleAction.new("Home", None)
        home_action.connect("activate", self.on_Home)
        app.add_action(home_action)

        imagecapture_action = Gio.SimpleAction.new("imagecapture", None)
        imagecapture_action.connect("activate", self.on_imagecapture)
        app.add_action(imagecapture_action)

        livecapture_action = Gio.SimpleAction.new("livecapture", None)
        livecapture_action.connect("activate", self.on_livecapture)
        app.add_action(livecapture_action)

        videocapture_action = Gio.SimpleAction.new("videocapture", None)
        videocapture_action.connect("activate", self.on_videocapture)
        app.add_action(videocapture_action)

        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", self.on_quit)
        app.add_action(quit_action)

    def on_Home(self, action, parameter):
        Home.Home()

    def on_imagecapture(self, action, parameter):
        imagecapture.imagecapture()

    def on_livecapture(self, action, parameter):
        livecapture.livecapture()
    def on_videocapture(self, action, parameter):
        videocapture.videocapture()

    def on_quit(self, action, parameter):
        self.get_application().quit()


class ObjectDetection(Gtk.Application):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect('activate', self.on_activate)

    def on_activate(self, app):
        self.win = ObjectDetectionExperimental(app)
        self.win.present()


if __name__ == "__main__":
    app = ObjectDetection(application_id='org.DeepActionExperimental.GtkApplication')
    app.run(sys.argv)
