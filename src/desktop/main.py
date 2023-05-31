#!/usr/bin/env python3
import sys
import os
import gi
import importlib
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
gi.require_version('Notify', '0.7')
from gi.repository import Gtk, Gio, Gdk, Notify

class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        menu = Gio.Menu.new()

        # Home menu
        home_menu = Gio.Menu.new()
        home_menu.append("Home", "win.home")
        home_menu_item = Gio.MenuItem.new("Home", "win.home")
        home_menu_item.set_detailed_action("win.home")
        menu.append_item(home_menu_item)

        # Satellite menu
        satellite_menu = Gio.Menu.new()
        satellite_menu.append("Satellite View", "win.satellite_view")
        satellite_menu_item = Gio.MenuItem.new("Satellite View", "win.satellite_view")
        satellite_menu_item.set_detailed_action("win.satellite_view")
        menu.append_item(satellite_menu_item)
        
        # Object View menu
        object_menu = Gio.Menu.new()
        object_menu.append("Object View", "win.object_view")
        object_menu_item = Gio.MenuItem.new("Object View", "win.object_view")
        object_menu_item.set_detailed_action("win.object_view")
        menu.append_item(object_menu_item)
        
        # Mobile View menu
        mobile_menu = Gio.Menu.new()
        mobile_menu.append("Mobile View", "win.mobile_view")
        mobile_menu_item = Gio.MenuItem.new("Mobile View", "win.mobile_view")
        mobile_menu_item.set_detailed_action("win.mobile_view")
        menu.append_item(mobile_menu_item)
        
        # Query menu
        query_menu = Gio.Menu.new()
        query_menu.append("Query", "win.query")
        query_menu_item = Gio.MenuItem.new("Query", "win.query")
        query_menu_item.set_detailed_action("win.query")
        menu.append_item(query_menu_item)

        self.popover = Gtk.PopoverMenu()
        self.popover.set_menu_model(menu)

        self.header = Gtk.HeaderBar()
        self.set_titlebar(self.header)

        self.hamburger = Gtk.MenuButton()
        self.hamburger.set_popover(self.popover)
        self.hamburger.set_icon_name("open-menu-symbolic")
        self.header.pack_start(self.hamburger)

        # Initialize the notification system
        Notify.init("Deep-Actions-Experimental")

        action = Gio.SimpleAction.new("home")
        action.connect("activate", self.home_menu)
        self.add_action(action)

        action = Gio.SimpleAction.new("satellite_view")
        action.connect("activate", self.satellite_menu)
        self.add_action(action)
        
        action = Gio.SimpleAction.new("object_view")
        action.connect("activate", self.object_menu)
        self.add_action(action)
        
        action = Gio.SimpleAction.new("mobile_view")
        action.connect("activate", self.mobile_menu)
        self.add_action(action)
        
        action = Gio.SimpleAction.new("query")
        action.connect("activate", self.query_menu)
        self.add_action(action)

    def home_menu(self, action, parameter):
        try:
            home_module = importlib.import_module("home")
            home_module.home()  # Assuming there is a function called 'home' in the 'home' module
        except ImportError:
            print("Failed to import 'home' module.")

    def satellite_menu(self, action, parameter):
        try:
            satellite_module = importlib.import_module("satellite")
            satellite_module.satellite()  # Assuming there is a function called 'satellite' in the 'satellite' module
        except ImportError:
            print("Failed to import 'satellite' module.")
            
    def object_menu(self, action, parameter):
        try:
            object_module = importlib.import_module("object")
            object_module.object()  # Assuming there is a function called 'object' in the 'object' module
        except ImportError:
            print("Failed to import 'Object' module.")
    
    def mobile_menu(self, action, parameter):
        try:
            mobile_module = importlib.import_module("mobile")
            mobile_module.mobile()  # Assuming there is a function called 'mobile' in the 'mobile' module
        except ImportError:
            print("Failed to import 'Mobile' module.")
    def query_menu(slef,action,paramenter):
        try:
            query_module = importlib.import_module("query")
            query_module.query()  # Assuming there is a function called 'query' in the 'query' module
        except ImportError:
            print("Failed to import 'Query' module.")
class MyApp(Gtk.Application):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect('activate', self.on_activate)

    def on_activate(self, app):
        self.win = MainWindow(application=app, title="Deep-Actions-Experimental")
        self.win.present()

if __name__ == "__main__":
    app = MyApp(application_id='org.PenetrationApp.GtkApplication')
    app.run(sys.argv)