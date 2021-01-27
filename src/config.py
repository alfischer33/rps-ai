

#imports postgres credentials
def config(filename='database.ini', section='postgresql'):
    
    try: 
        from configparser import ConfigParser
    
        # create a parser
        parser = ConfigParser()
        # read config file
        parser.read(filename)

        # get section, default to postgresql
        db = {}
        
        # Checks to see if section (postgresql) parser exists
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
            
        # Returns an error if a parameter is called that is not listed in the initialization file
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    except:
        import os

        db = {}
        db['user'] = os.environ['user']
        db['password'] = os.environ['password']
        db['host'] = os.environ['host']
        db['port'] = os.environ['port']
        db['database'] = os.environ['database']
    
    return db