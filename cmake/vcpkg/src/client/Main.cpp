#include <thread>
#include <iostream>

#include <SFML/Graphics.hpp>

int main()
{
    sf::RenderWindow windows(sf::VideoMode(640, 480), "SFML Application");
    sf::CircleShape shape;
    shape.setRadius(40.f);
    shape.setPosition(100.f, 100.f);
    shape.setFillColor(sf::Color::Cyan);
    while (windows.isOpen()) {
        sf::Event event;
        while (windows.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                windows.close();
        }
        windows.clear();
        windows.draw(shape);
        windows.display();
    }
    return 0;
}
