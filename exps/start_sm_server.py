from ravt.launchers import SharedMemoryLauncher


def main():
    server = SharedMemoryLauncher()
    server.run()


if __name__ == '__main__':
    main()
